import pytest
from unittest import mock
import requests
import json
import os
import sys
server_path = os.path.abspath('model_training/Server')
if server_path not in sys.path:
    sys.path.insert(0, server_path)
from inference_service.backend_app import app as flask_app

@pytest.fixture
def app():
    """Create a test Flask app with the routes from backend_app."""
    app = flask_app
    app.config.update({
        "TESTING": True,
        "ML_HOST": "test-ml-host",
        "ML_PORT": 8083
    })
    yield app

@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()

@pytest.fixture
def mock_requests_get():
    """Create a mock for requests.get."""
    with mock.patch('backend_app.requests.get') as mock_get:
        yield mock_get

def test_home_route(client):
    """Test the home route returns correct message."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Movie Recommendation Service is Running!" in response.data

def test_recommend_success(client, mock_requests_get):
    """Test successful recommendation request."""
    # Setup mock response
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = ["movie1", "movie2", "movie3"]
    mock_requests_get.return_value = mock_response
    
    # Make request to backend service
    response = client.get('/recommend/123')
    
    # Check response
    assert response.status_code == 200
    assert response.data == b"movie1,movie2,movie3"
    
    # Verify request was made to ML service
    mock_requests_get.assert_called_once()
    assert "test-ml-host:8083/recommend/123" in mock_requests_get.call_args[0][0]

def test_recommend_ml_service_error(client, mock_requests_get):
    """Test handling ML service error."""
    # Setup mock response with error
    mock_response = mock.Mock()
    mock_response.status_code = 500
    mock_requests_get.return_value = mock_response
    
    # Make request to backend service
    response = client.get('/recommend/123')
    
    # Check response
    assert response.status_code == 500
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "ML service error: 500" in response_data["error"]

def test_recommend_timeout(client, mock_requests_get):
    """Test handling timeout from ML service."""
    # Setup mock to raise timeout
    mock_requests_get.side_effect = requests.exceptions.Timeout("Connection timed out")
    
    # Make request to backend service
    response = client.get('/recommend/123')
    
    # Check response
    assert response.status_code == 504
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "ML service timed out" in response_data["error"]

def test_recommend_connection_error(client, mock_requests_get):
    """Test handling connection error to ML service."""
    # Setup mock to raise connection error
    mock_requests_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
    
    # Make request to backend service
    response = client.get('/recommend/123')
    
    # Check response
    assert response.status_code == 500
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "Connection refused" in response_data["error"]

def test_recommend_empty_list(client, mock_requests_get):
    """Test handling empty movie list from ML service."""
    # Setup mock response with empty list
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_requests_get.return_value = mock_response
    
    # Make request to backend service
    response = client.get('/recommend/123')
    
    # Check response
    assert response.status_code == 200
    assert response.data == b""  # Empty string for empty list

def test_recommend_invalid_json(client, mock_requests_get):
    """Test handling invalid JSON from ML service."""
    # Setup mock response with invalid JSON
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_requests_get.return_value = mock_response
    
    # Make request to backend service
    response = client.get('/recommend/123')
    
    # Check response
    assert response.status_code == 500
    response_data = json.loads(response.data)
    assert "error" in response_data

def test_config_loading():
    """Test that app loads configuration correctly."""
    with mock.patch('inference_service.backend_app.app.config.from_object') as mock_config:
        # Mock app.run to avoid actually starting the server
        with mock.patch('backend_app.app.run'):
            # Import __main__ code
            with mock.patch('backend_app.__name__', '__main__'):
                # Execute the main block
                exec(open('inference_service/backend_app.py').read())
                
                # Verify config was loaded
                mock_config.assert_called_once_with('config.Config')

def test_app_run_parameters():
    """Test that app runs with correct parameters from config."""
    with mock.patch('inference_service.backend_app.app.config.from_object'):
        # Setup mock config values
        test_config = {
            'HOST': '127.0.0.1',
            'PORT': 9999,
            'DEBUG': True
        }
        
        # Mock app.config.get to return our test values
        def mock_get(key, default):
            return test_config.get(key, default)
        
        with mock.patch('inference_service.backend_app.app.config.get', side_effect=mock_get):
            # Mock app.run to capture parameters
            with mock.patch('inference_service.backend_app.app.run') as mock_run:
                # Import __main__ code
                with mock.patch('inference_service.backend_app.__name__', '__main__'):
                    # Execute the main block
                    exec(open('inference_service/backend_app.py').read())
                    
                    # Verify app.run was called with correct parameters
                    mock_run.assert_called_once_with('127.0.0.1', 9999, True)