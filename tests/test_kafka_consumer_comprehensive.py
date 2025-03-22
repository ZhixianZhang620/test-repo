import pytest
import datetime
from unittest import mock
import json
from datetime import datetime as dt

# We need to mock kafka_consumer module, assuming it's in data_processing package
# If it's in a different location, update the import path
class MockKafkaConsumer:
    """Mock class for testing without actual Kafka connection."""
    def __init__(self, *args, **kwargs):
        self.messages = []
    
    def __iter__(self):
        return iter(self.messages)
    
    def set_messages(self, messages):
        """Set messages to be consumed by the iterator."""
        self.messages = [mock.Mock(value=m.encode()) for m in messages]

@pytest.fixture
def mock_db():
    """Create a mock database for testing."""
    db_mock = mock.Mock()
    
    # Setup collections
    db_mock.user_info = mock.Mock()
    db_mock.movie_info = mock.Mock()
    db_mock.user_rate = mock.Mock()
    db_mock.user_watch = mock.Mock()
    db_mock.recommendation_log = mock.Mock()
    
    # Setup return values
    db_mock.user_info.update_one.return_value = None
    db_mock.movie_info.update_one.return_value = None
    db_mock.user_rate.insert_one.return_value = None
    db_mock.user_watch.insert_one.return_value = None
    db_mock.recommendation_log.insert_one.return_value = None
    
    return db_mock

@pytest.fixture
def mock_requests():
    """Create a mock for requests."""
    with mock.patch('requests.get') as mock_get:
        # Setup mock response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "user123",
            "age": 30,
            "occupation": "Engineer",
            "gender": "M",
            "movie_id": "movie456",
            "adult": False,
            "genres": [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}],
            "release_date": "2023-01-01",
            "original_language": "en"
        }
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture
def mock_kafka_consumer():
    """Create a mock for KafkaConsumer."""
    mock_consumer = MockKafkaConsumer()
    with mock.patch('data_processing.kafka_consumer.KafkaConsumer', return_value=mock_consumer):
        yield mock_consumer

@pytest.fixture
def kafka_consumer_app(mock_db, mock_kafka_consumer):
    """Create a KafkaConsumerApp instance with mocks."""
    with mock.patch('data_processing.kafka_consumer.DB', return_value=mock_db):
        # Import needs to be here after mocking
        from data_processing.kafka_consumer import KafkaConsumerApp
        
        app = KafkaConsumerApp(topic='test_topic', bootstrap_servers=['localhost:9092'])
        app.DB = mock_db
        app.consumer = mock_kafka_consumer
        yield app

def test_process_user_info_valid(kafka_consumer_app, mock_requests):
    """Test processing valid user info."""
    # Setup request to return user info
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "user_id": "user123",
        "age": 30,
        "occupation": "Engineer",
        "gender": "M"
    }
    mock_requests.return_value = mock_response
    
    # Call the method
    kafka_consumer_app.process_user_info("user123")
    
    # Verify API call
    mock_requests.assert_called_once()
    assert "user123" in mock_requests.call_args[0][0]
    
    # Verify DB update
    kafka_consumer_app.DB.user_info.update_one.assert_called_once()
    update_filter = kafka_consumer_app.DB.user_info.update_one.call_args[0][0]
    update_data = kafka_consumer_app.DB.user_info.update_one.call_args[0][1]["$set"]
    assert update_filter == {"user_id": "user123"}
    assert update_data["age"] == 30
    assert update_data["occupation"] == "Engineer"
    assert update_data["gender"] == "M"

def test_process_user_info_api_error(kafka_consumer_app, mock_requests):
    """Test handling API errors in user info processing."""
    # Setup request to return error
    mock_requests.return_value.status_code = 404
    
    # Call the method - should not raise exception
    kafka_consumer_app.process_user_info("user123")
    
    # Verify API call
    mock_requests.assert_called_once()
    
    # Verify DB was not updated
    kafka_consumer_app.DB.user_info.update_one.assert_not_called()

def test_process_user_info_db_error(kafka_consumer_app, mock_requests):
    """Test handling DB errors in user info processing."""
    # Setup DB to raise exception
    kafka_consumer_app.DB.user_info.update_one.side_effect = Exception("DB error")
    
    # Call the method - should not raise exception
    kafka_consumer_app.process_user_info("user123")
    
    # Verify API call was made
    mock_requests.assert_called_once()

def test_process_movie_info_valid(kafka_consumer_app, mock_requests):
    """Test processing valid movie info."""
    # Setup request to return movie info
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "movie_id": "movie456",
        "adult": False,
        "genres": [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}],
        "release_date": "2023-01-01",
        "original_language": "en"
    }
    mock_requests.return_value = mock_response
    
    # Call the method
    kafka_consumer_app.process_movie_info("movie456")
    
    # Verify API call
    mock_requests.assert_called_once()
    assert "movie456" in mock_requests.call_args[0][0]
    
    # Verify DB update
    kafka_consumer_app.DB.movie_info.update_one.assert_called_once()
    update_filter = kafka_consumer_app.DB.movie_info.update_one.call_args[0][0]
    update_data = kafka_consumer_app.DB.movie_info.update_one.call_args[0][1]["$set"]
    assert update_filter == {"movie_id": "movie456"}
    assert update_data["adult"] is False
    assert len(update_data["genres"]) == 2
    assert update_data["original_language"] == "en"
    assert isinstance(update_data["release_date"], dt)

def test_process_movie_info_api_error(kafka_consumer_app, mock_requests):
    """Test handling API errors in movie info processing."""
    # Setup request to return error
    mock_requests.return_value.status_code = 404
    
    # Call the method - should not raise exception
    kafka_consumer_app.process_movie_info("movie456")
    
    # Verify API call
    mock_requests.assert_called_once()
    
    # Verify DB was not updated
    kafka_consumer_app.DB.movie_info.update_one.assert_not_called()

def test_process_movie_info_invalid_date(kafka_consumer_app, mock_requests):
    """Test handling invalid release dates in movie info."""
    # Setup request to return movie with invalid date
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "movie_id": "movie456",
        "adult": False,
        "genres": [{"id": 28, "name": "Action"}],
        "release_date": "invalid-date",
        "original_language": "en"
    }
    mock_requests.return_value = mock_response
    
    # Call the method - should handle invalid date
    kafka_consumer_app.process_movie_info("movie456")
    
    # Verify DB update still occurred
    kafka_consumer_app.DB.movie_info.update_one.assert_called_once()

def test_process_user_rate_valid(kafka_consumer_app):
    """Test processing valid user rating message."""
    # Set up the mock methods
    kafka_consumer_app.process_user_info = mock.Mock()
    kafka_consumer_app.process_movie_info = mock.Mock()
    
    # Call with valid rate message
    rate_msg = "2023-01-01T12:00:00,user123,GET /rate/movie_id=movie456=4.5"
    kafka_consumer_app.process_user_rate(rate_msg)
    
    # Verify user and movie info were processed
    kafka_consumer_app.process_user_info.assert_called_once_with("user123")
    kafka_consumer_app.process_movie_info.assert_called_once_with("movie456")
    
    # Verify rating was inserted to DB
    kafka_consumer_app.DB.user_rate.insert_one.assert_called_once()
    inserted_data = kafka_consumer_app.DB.user_rate.insert_one.call_args[0][0]
    assert inserted_data["user_id"] == "user123"
    assert inserted_data["movie_id"] == "movie456"
    assert inserted_data["score"] == 4.5
    assert isinstance(inserted_data["time"], dt)

def test_process_user_rate_invalid_format(kafka_consumer_app):
    """Test processing invalid user rating message format."""
    # Set up the mock methods
    kafka_consumer_app.process_user_info = mock.Mock()
    kafka_consumer_app.process_movie_info = mock.Mock()
    
    # Call with invalid rate message
    invalid_msg = "2023-01-01T12:00:00,user123,GET /rate/invalid_format"
    kafka_consumer_app.process_user_rate(invalid_msg)
    
    # Verify no processing occurred
    kafka_consumer_app.process_user_info.assert_not_called()
    kafka_consumer_app.process_movie_info.assert_not_called()
    kafka_consumer_app.DB.user_rate.insert_one.assert_not_called()

def test_process_user_rate_invalid_score(kafka_consumer_app):
    """Test processing user rating with invalid score value."""
    # Set up the mock methods
    kafka_consumer_app.process_user_info = mock.Mock()
    kafka_consumer_app.process_movie_info = mock.Mock()
    
    # Call with invalid score
    invalid_score_msg = "2023-01-01T12:00:00,user123,GET /rate/movie_id=movie456=invalid"
    kafka_consumer_app.process_user_rate(invalid_score_msg)
    
    # Verify no DB insert occurred
    kafka_consumer_app.DB.user_rate.insert_one.assert_not_called()

def test_process_user_watch_history_valid(kafka_consumer_app):
    """Test processing valid user watch history message."""
    # Set up the mock methods
    kafka_consumer_app.process_user_info = mock.Mock()
    kafka_consumer_app.process_movie_info = mock.Mock()
    
    # Call with valid watch message
    watch_msg = "2023-01-01T13:00:00,user123,GET /data/m/movie456/120"
    kafka_consumer_app.process_user_watch_history(watch_msg)
    
    # Verify user and movie info were processed
    kafka_consumer_app.process_user_info.assert_called_once_with("user123")
    kafka_consumer_app.process_movie_info.assert_called_once_with("movie456")
    
    # Verify watch data was inserted to DB
    kafka_consumer_app.DB.user_watch.insert_one.assert_called_once()
    inserted_data = kafka_consumer_app.DB.user_watch.insert_one.call_args[0][0]
    assert inserted_data["user_id"] == "user123"
    assert inserted_data["movie_id"] == "movie456"
    assert inserted_data["minute_mpg"] == "120"
    assert isinstance(inserted_data["time"], dt)

def test_process_user_watch_history_invalid_format(kafka_consumer_app):
    """Test processing invalid user watch history message format."""
    # Set up the mock methods
    kafka_consumer_app.process_user_info = mock.Mock()
    kafka_consumer_app.process_movie_info = mock.Mock()
    
    # Call with invalid watch message
    invalid_msg = "2023-01-01T13:00:00,user123,GET /data/invalid_format"
    kafka_consumer_app.process_user_watch_history(invalid_msg)
    
    # Verify no processing occurred
    kafka_consumer_app.process_user_info.assert_not_called()
    kafka_consumer_app.process_movie_info.assert_not_called()
    kafka_consumer_app.DB.user_watch.insert_one.assert_not_called()

def test_process_recommendation_result_valid(kafka_consumer_app):
    """Test processing valid recommendation result message."""
    # Call with valid recommendation message
    rec_msg = "2023-01-01T14:00:00, user123, recommendation request, status 200, result: movie1, movie2, 150ms"
    kafka_consumer_app.process_recommendation_result(rec_msg)
    
    # Verify recommendation was logged
    kafka_consumer_app.DB.recommendation_log.insert_one.assert_called_once()
    inserted_data = kafka_consumer_app.DB.recommendation_log.insert_one.call_args[0][0]
    assert inserted_data["user_id"] == "user123"
    assert inserted_data["status_code"] == 200
    assert inserted_data["recommendation_results"] == ["movie1", "movie2"]
    assert inserted_data["response_time"] == 150
    assert isinstance(inserted_data["time"], dt)

def test_process_recommendation_result_non_200_status(kafka_consumer_app):
    """Test processing recommendation result with non-200 status code."""
    # Call with error status recommendation message
    error_msg = "2023-01-01T14:00:00, user123, recommendation request, status 500, result: error, 150ms"
    kafka_consumer_app.process_recommendation_result(error_msg)
    
    # Verify no log entry was created
    kafka_consumer_app.DB.recommendation_log.insert_one.assert_not_called()

def test_process_recommendation_result_invalid_format(kafka_consumer_app):
    """Test processing recommendation result with invalid format."""
    # Call with invalid recommendation message
    invalid_msg = "2023-01-01T14:00:00, user123, invalid format"
    kafka_consumer_app.process_recommendation_result(invalid_msg)
    
    # Verify no log entry was created
    kafka_consumer_app.DB.recommendation_log.insert_one.assert_not_called()

def test_consume_messages_valid(kafka_consumer_app, mock_kafka_consumer):
    """Test consuming valid messages from Kafka."""
    # Setup mock methods
    kafka_consumer_app.process_user_rate = mock.Mock()
    kafka_consumer_app.process_user_watch_history = mock.Mock()
    kafka_consumer_app.process_recommendation_result = mock.Mock()
    
    # Setup messages
    mock_kafka_consumer.set_messages([
        "2023-01-01T12:00:00,user123,GET /rate/movie_id=movie456=4.5",
        "2023-01-01T13:00:00,user123,GET /data/m/movie456/120",
        "2023-01-01T14:00:00, user123, recommendation request, status 200, result: movie1, movie2, 150ms"
    ])
    
    # Call consume_messages
    kafka_consumer_app.consume_messages()
    
    # Verify each message was processed
    assert kafka_consumer_app.process_user_rate.call_count == 1
    assert kafka_consumer_app.process_user_watch_history.call_count == 1
    assert kafka_consumer_app.process_recommendation_result.call_count == 1

def test_consume_messages_with_exceptions(kafka_consumer_app, mock_kafka_consumer):
    """Test consuming messages with exceptions during processing."""
    # Setup mock methods to raise exceptions
    kafka_consumer_app.process_user_rate = mock.Mock(side_effect=Exception("Rate error"))
    kafka_consumer_app.process_user_watch_history = mock.Mock()
    kafka_consumer_app.process_recommendation_result = mock.Mock()
    
    # Setup messages
    mock_kafka_consumer.set_messages([
        "2023-01-01T12:00:00,user123,GET /rate/movie_id=movie456=4.5",
        "2023-01-01T13:00:00,user123,GET /data/m/movie456/120"
    ])
    
    # Call consume_messages - should not crash
    kafka_consumer_app.consume_messages()
    
    # Verify exception didn't stop processing
    assert kafka_consumer_app.process_user_rate.call_count == 1
    assert kafka_consumer_app.process_user_watch_history.call_count == 1

def test_unrecognized_message_format(kafka_consumer_app, mock_kafka_consumer):
    """Test handling unrecognized message formats."""
    # Setup mock methods
    kafka_consumer_app.process_user_rate = mock.Mock()
    kafka_consumer_app.process_user_watch_history = mock.Mock()
    kafka_consumer_app.process_recommendation_result = mock.Mock()
    
    # Setup unrecognized message
    mock_kafka_consumer.set_messages([
        "2023-01-01T12:00:00,user123,UNRECOGNIZED FORMAT"
    ])
    
    # Call consume_messages - should not crash
    kafka_consumer_app.consume_messages()
    
    # Verify no processing methods were called
    assert kafka_consumer_app.process_user_rate.call_count == 0
    assert kafka_consumer_app.process_user_watch_history.call_count == 0
    assert kafka_consumer_app.process_recommendation_result.call_count == 0

def test_message_with_multiple_matches(kafka_consumer_app, mock_kafka_consumer):
    """Test handling messages that match multiple patterns."""
    # Setup mock methods
    kafka_consumer_app.process_user_rate = mock.Mock()
    kafka_consumer_app.process_user_watch_history = mock.Mock()
    
    # Setup message that could match both rate and watch patterns
    mock_kafka_consumer.set_messages([
        "2023-01-01T12:00:00,user123,GET /rate/movie_id=movie456=4.5 /data/m/movie789/120"
    ])
    
    # Call consume_messages
    kafka_consumer_app.consume_messages()
    
    # Verify only the first matching pattern was processed
    assert kafka_consumer_app.process_user_rate.call_count == 1
    assert kafka_consumer_app.process_user_watch_history.call_count == 0