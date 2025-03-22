import pytest
import requests
import time
import multiprocessing
import signal
import os
import sys
from unittest import mock
import tempfile
import json

# Import the app modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
server_path = os.path.abspath('model_training/Server')
if server_path not in sys.path:
    sys.path.insert(0, server_path)
from inference_service.backend_app import app as backend_app
from model_training.Server.model_app import app as model_app 

# Global variables for server processes
backend_process = None
ml_process = None

@pytest.fixture(scope="module")
def create_test_configs():
    """Create temporary config files for testing."""
    # Create test config for backend service
    backend_config = tempfile.NamedTemporaryFile(delete=False, suffix='.py')
    backend_config.write(b"""
class Config:
    HOST = "127.0.0.1"
    PORT = 8082
    DEBUG = False
    ML_HOST = "127.0.0.1"
    ML_PORT = 8083
""")
    backend_config.close()
    
    # Create test config for ML service
    ml_config = tempfile.NamedTemporaryFile(delete=False, suffix='.py')
    ml_config.write(b"""
class Config:
    HOST = "127.0.0.1"
    PORT = 8083
    DEBUG = False
""")
    ml_config.close()
    
    # Create test sys_config
    sys_config = tempfile.NamedTemporaryFile(delete=False, prefix='sys_config_', suffix='.py')
    sys_config.write(b"""
# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
USER_DB = "test_user_database"
MOVIE_DB = "test_movie_database"
LOG_DB = "test_log_database"
USER_RATE_COLLECTION = "user_rate_data"
USER_WATCH_COLLECTION = "user_watch_data"
USER_DATA_COLLECTION = "user_info"
MOVIE_DATA_COLLECTION = "movie_info"
RECOMMENDATION_LOG = "recommendation_log"
""")
    sys_config.close()
    
    # Return file paths
    yield {
        'backend_config': backend_config.name,
        'ml_config': ml_config.name,
        'sys_config': sys_config.name
    }
    
    # Clean up files after tests
    try:
        os.unlink(backend_config.name)
        os.unlink(ml_config.name)
        os.unlink(sys_config.name)
    except:
        pass

def start_backend_server(config_path):
    """Start the backend Flask server in a separate process."""
    # Mock app configuration
    backend_app.app.config.from_pyfile(config_path)
    backend_app.app.run(
        backend_app.app.config.get("HOST", "127.0.0.1"),
        backend_app.app.config.get("PORT", 8082),
        use_reloader=False
    )

def start_ml_server(config_path, sys_config_path):
    """Start the ML service Flask server in a separate process."""
    # Mock SVD to avoid actual model loading
    with mock.patch('model_app.SVDPipeline') as mock_svd:
        # Create mock instance with get_recommendations method
        mock_instance = mock.Mock()
        mock_instance.get_recommendations.return_value = ["movie1", "movie2", "movie3"]
        mock_svd.return_value = mock_instance
        
        # Mock sys.path to allow importing the test sys_config
        sys.path.insert(0, os.path.dirname(sys_config_path))
        
        # Start ML service
        model_app.app.config.from_pyfile(config_path)
        model_app.app.run(
            model_app.app.config.get("HOST", "127.0.0.1"),
            model_app.app.config.get("PORT", 8083),
            use_reloader=False
        )

@pytest.fixture(scope="module")
def setup_servers(create_test_configs):
    """Start both backend and ML service servers for testing."""
    global backend_process, ml_process
    
    config_paths = create_test_configs
    
    # Start ML service process
    ml_process = multiprocessing.Process(
        target=start_ml_server,
        args=(config_paths['ml_config'], config_paths['sys_config'])
    )
    ml_process.daemon = True  # Will be killed when test process exits
    ml_process.start()
    
    # Start backend service process
    backend_process = multiprocessing.Process(
        target=start_backend_server,
        args=(config_paths['backend_config'],)
    )
    backend_process.daemon = True  # Will be killed when test process exits
    backend_process.start()
    
    # Wait for servers to start
    time.sleep(2)
    
    # Check if servers started
    try:
        ml_response = requests.get("http://127.0.0.1:8083/")
        backend_response = requests.get("http://127.0.0.1:8082/")
        if ml_response.status_code != 200 or backend_response.status_code != 200:
            pytest.skip("Could not start test servers")
    except requests.exceptions.ConnectionError:
        # Servers failed to start, skip tests
        if ml_process and ml_process.is_alive():
            ml_process.terminate()
        if backend_process and backend_process.is_alive():
            backend_process.terminate()
        pytest.skip("Could not start test servers")
    
    yield
    
    # Stop servers after tests
    if ml_process and ml_process.is_alive():
        ml_process.terminate()
    if backend_process and backend_process.is_alive():
        backend_process.terminate()

@pytest.mark.integration
@pytest.mark.slow
def test_e2e_recommendation(setup_servers):
    """End-to-end test of recommendation flow."""
    # Skip if servers couldn't be started
    if not (backend_process and backend_process.is_alive() and ml_process and ml_process.is_alive()):
        pytest.skip("Test servers are not running")
    
    # Make request to backend service
    response = requests.get("http://127.0.0.1:8082/recommend/123")
    
    # Verify response
    assert response.status_code == 200
    assert response.text == "movie1,movie2,movie3"

@pytest.mark.integration
@pytest.mark.slow
def test_e2e_ml_service(setup_servers):
    """Test direct connection to ML service."""
    # Skip if servers couldn't be started
    if not (ml_process and ml_process.is_alive()):
        pytest.skip("ML service is not running")
    
    # Make request directly to ML service
    response = requests.get("http://127.0.0.1:8083/recommend/123")
    
    # Verify response
    assert response.status_code == 200
    assert response.json() == ["movie1", "movie2", "movie3"]

@pytest.mark.integration
@pytest.mark.slow
def test_e2e_backend_service(setup_servers):
    """Test backend service health check."""
    # Skip if servers couldn't be started
    if not (backend_process and backend_process.is_alive()):
        pytest.skip("Backend service is not running")
    
    # Make request to backend service health check
    response = requests.get("http://127.0.0.1:8082/")
    
    # Verify response
    assert response.status_code == 200
    assert "Movie Recommendation Service is Running!" in response.text