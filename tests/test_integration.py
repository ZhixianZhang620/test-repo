import pytest
import json
import requests
from unittest import mock
import subprocess
import time
import signal
import os
from flask import Flask

# Define server process variable
server_process = None

@pytest.fixture(scope="module")
def setup_test_database():
    """Setup a test database for integration testing."""
    with mock.patch('pymongo.MongoClient') as mock_client:
        # Create mock collections
        mock_db = mock.Mock()
        mock_movie_db = mock.Mock()
        mock_user_db = mock.Mock()
        
        # Setup movie_info collection
        mock_movie_info = mock.Mock()
        mock_movie_info.find.return_value = iter([
            {
                'movie_id': '1',
                'adult': False,
                'genres': [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}],
                'release_date': '2023-01-01',
                'original_language': 'en'
            },
            {
                'movie_id': '2',
                'adult': False,
                'genres': [{'id': 35, 'name': 'Comedy'}],
                'release_date': '2023-02-01',
                'original_language': 'en'
            }
        ])
        
        # Setup user_rate_data collection
        mock_rate_data = mock.Mock()
        mock_rate_data.find.return_value = iter([
            {'user_id': 'user1', 'movie_id': '1', 'score': 4.5},
            {'user_id': 'user1', 'movie_id': '2', 'score': 3.0},
            {'user_id': 'user2', 'movie_id': '1', 'score': 5.0}
        ])
        
        # Setup user_watch_data collection
        mock_watch_data = mock.Mock()
        mock_watch_data.find.return_value = iter([
            {'user_id': 'user1', 'movie_id': '3', 'minute_mpg': '120'},
            {'user_id': 'user2', 'movie_id': '2', 'minute_mpg': '90'}
        ])
        
        # Setup user_info collection
        mock_user_info = mock.Mock()
        mock_user_info.find.return_value = iter([
            {'user_id': 'user1', 'age': 30, 'occupation': 'Engineer', 'gender': 'M'},
            {'user_id': 'user2', 'age': 25, 'occupation': 'Student', 'gender': 'F'}
        ])
        
        # Assign collections to databases
        mock_movie_db.__getitem__.side_effect = lambda x: {
            'movie_info': mock_movie_info,
            'user_rate_data': mock_rate_data,
            'user_watch_data': mock_watch_data
        }.get(x, mock.Mock())
        
        mock_user_db.__getitem__.side_effect = lambda x: {
            'user_info': mock_user_info
        }.get(x, mock.Mock())
        
        # Assign databases to client
        mock_client.return_value.__getitem__.side_effect = lambda x: {
            'movie_database': mock_movie_db,
            'user_database': mock_user_db,
        }.get(x, mock.Mock())
        
        yield

@pytest.fixture(scope="module")
def start_server():
    """Start the Flask server for testing and stop it after tests complete."""
    global server_process
    
    # Create a temporary config file for testing
    with open('test_config.py', 'w') as f:
        f.write("""
class Config:
    HOST = '127.0.0.1'
    PORT = 8084
    DEBUG = False
""")
    
    # Mock the sys_config.py file
    with open('test_sys_config.py', 'w') as f:
        f.write("""
# MongoDB Configuration
MONGO_URI = ""
USER_DB = 'user_database'
MOVIE_DB = 'movie_database'
LOG_DB = 'log_database'
USER_RATE_COLLECTION = 'user_rate_data'
USER_WATCH_COLLECTION = 'user_watch_data'
USER_DATA_COLLECTION = 'user_info'
MOVIE_DATA_COLLECTION = 'movie_info'
RECOMMENDATION_LOG = 'recommendation_log'
""")

    # Start server with mocked config
    # We'll use Flask's built-in test client instead of a real server for integration tests
    with mock.patch('sys.argv', ['model_app.py']):
        with mock.patch('sys.path', ['.']):
            with mock.patch('model_app.app.config.from_object', side_effect=lambda _: None):
                with mock.patch('model_app.SVDPipeline') as mock_svd:
                    # Setup mock SVD
                    mock_instance = mock.Mock()
                    mock_instance.get_recommendations.return_value = ['movie1', 'movie2', 'movie3']
                    mock_svd.return_value = mock_instance
                    
                    # Import app after mocking
                    from model_app import app
                    
                    # Configure test client
                    app.config['TESTING'] = True
                    client = app.test_client()
                    
                    yield client
    
    # Clean up temporary files
    try:
        os.remove('test_config.py')
        os.remove('test_sys_config.py')
    except:
        pass

@pytest.mark.integration
def test_server_health(start_server):
    """Test that the server is running and health check endpoint works."""
    client = start_server
    response = client.get('/')
    assert response.status_code == 200
    assert b"ML Service is Running!" in response.data

@pytest.mark.integration
def test_recommendation_endpoint(start_server, setup_test_database):
    """Test the recommendation endpoint returns expected results."""
    client = start_server
    
    # Test for valid user
    response = client.get('/recommend/123')
    assert response.status_code == 200
    
    # Verify response format
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Verify each movie ID is a string
    for movie_id in data:
        assert isinstance(movie_id, str)

@pytest.mark.integration
def test_recommendation_error_handling(start_server):
    """Test error handling in the recommendation endpoint."""
    client = start_server
    
    # Mock SVDPipeline to raise exception
    with mock.patch('model_app.svd_pipeline.get_recommendations', 
                   side_effect=Exception("Integration test error")):
        response = client.get('/recommend/999')
        
        # Verify error response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data
        assert "Integration test error" in data["error"]