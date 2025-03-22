import pytest
import pandas as pd
import numpy as np
from unittest import mock
import datetime
import os
import sys
server_path = os.path.abspath('model_training/Server')
if server_path not in sys.path:
    sys.path.insert(0, server_path)
from model_training.Server.SVD import SVDPipeline, DB
import json

@pytest.fixture
def user_rate_data():
    """Sample user rating data for testing."""
    return [
        {
            "user_id": "user1",
            "movie_id": "1",
            "score": 5.0,
            "timestamp": datetime.datetime(2023, 1, 1, 10, 0, 0)
        },
        {
            "user_id": "user1",
            "movie_id": "2",
            "score": 3.5,
            "timestamp": datetime.datetime(2023, 1, 2, 10, 0, 0)
        },
        {
            "user_id": "user2",
            "movie_id": "1",
            "score": 4.0,
            "timestamp": datetime.datetime(2023, 1, 3, 10, 0, 0)
        },
        {
            "user_id": "user2",
            "movie_id": "3",
            "score": 3.0,
            "timestamp": datetime.datetime(2023, 1, 4, 10, 0, 0)
        }
    ]

@pytest.fixture
def user_watch_data():
    """Sample user watch history data for testing."""
    return [
        {
            "user_id": "user1",
            "movie_id": "3",
            "minute_mpg": "120",
            "timestamp": datetime.datetime(2023, 1, 5, 10, 0, 0)
        },
        {
            "user_id": "user2",
            "movie_id": "2",
            "minute_mpg": "90",
            "timestamp": datetime.datetime(2023, 1, 6, 10, 0, 0)
        },
        {
            "user_id": "user3",
            "movie_id": "1",
            "minute_mpg": "60",
            "timestamp": datetime.datetime(2023, 1, 7, 10, 0, 0)
        }
    ]

@pytest.fixture
def movie_data():
    """Sample movie data for testing."""
    return [
        {
            "movie_id": "1",
            "adult": False,
            "genres": [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}],
            "release_date": "2023-01-01",
            "original_language": "en"
        },
        {
            "movie_id": "2",
            "adult": False,
            "genres": [{"id": 35, "name": "Comedy"}],
            "release_date": "2023-02-01",
            "original_language": "en"
        },
        {
            "movie_id": "3",
            "adult": False,
            "genres": [{"id": 18, "name": "Drama"}],
            "release_date": "2023-03-01",
            "original_language": "en"
        },
        {
            "movie_id": "4",
            "adult": False,
            "genres": [{"id": 27, "name": "Horror"}],
            "release_date": "2023-04-01",
            "original_language": "en"
        },
        {
            "movie_id": "5",
            "adult": False,
            "genres": [{"id": 878, "name": "Science Fiction"}],
            "release_date": "2023-05-01",
            "original_language": "en"
        }
    ]

@pytest.fixture
def user_data():
    """Sample user data for testing."""
    return [
        {
            "user_id": "user1",
            "age": 30,
            "occupation": "Engineer",
            "gender": "M"
        },
        {
            "user_id": "user2",
            "age": 25,
            "occupation": "Student",
            "gender": "F"
        },
        {
            "user_id": "user3",
            "age": 40,
            "occupation": "Teacher",
            "gender": "F"
        }
    ]

def test_combined_ratings_df_creation(movie_data, user_rate_data, user_watch_data):
    """Test that combined ratings dataframe is created correctly."""
    # Mock database and collections
    with mock.patch('SVD.DB') as MockDB:
        # Setup mock DB
        mock_db = mock.Mock()
        mock_db.movie_db = mock.Mock()
        mock_db.user_db = mock.Mock()
        MockDB.return_value = mock_db
        
        # Setup movie_info collection
        mock_db.movie_db["movie_info"].find.return_value.limit.return_value = movie_data
        
        # Setup user_rate_data collection
        mock_db.movie_db["user_rate_data"].find.return_value.limit.return_value = user_rate_data
        
        # Setup user_watch_data collection
        mock_db.movie_db["user_watch_data"].find.return_value.limit.return_value = user_watch_data
        
        # Setup user_info collection
        mock_db.user_db["user_info"].find.return_value.limit.return_value = []
        
        # Create SVDPipeline with mocked data
        with mock.patch('SVD.SVD'):
            with mock.patch('SVD.Dataset'):
                with mock.patch('SVD.train_test_split'):
                    with mock.patch('SVD.Reader'):
                        pipeline = SVDPipeline()
                        pipeline.load_clean_data()
                        
                        # Test that ratings dataframe contains the correct columns
                        assert 'user_id' in pipeline.ratings_df.columns
                        assert 'movie_id' in pipeline.ratings_df.columns
                        assert 'score' in pipeline.ratings_df.columns
                        
                        # Test that watch dataframe contains the correct columns
                        assert 'user_id' in pipeline.watch_df.columns
                        assert 'movie_id' in pipeline.watch_df.columns
                        assert 'rating' in pipeline.watch_df.columns
                        
                        # Test that combined dataframe merges ratings and watch data
                        pipeline.ratings_df.rename(columns={'score': 'rating'}, inplace=True)
                        combined_df = pd.concat([pipeline.ratings_df, pipeline.watch_df], ignore_index=True)
                        
                        # Verify that all user IDs are in the combined dataframe
                        user_ids = set(r['user_id'] for r in user_rate_data).union(
                                    set(w['user_id'] for w in user_watch_data))
                        assert set(combined_df['user_id'].unique()) == user_ids
                        
                        # Verify all movie IDs are in the combined dataframe
                        movie_ids_in_activity = set(r['movie_id'] for r in user_rate_data).union(
                                                set(w['movie_id'] for w in user_watch_data))
                        assert set(combined_df['movie_id'].unique()) == movie_ids_in_activity

def test_data_cleaning_watch_time_normalization(user_watch_data):
    """Test that watch time is properly normalized to rating scale."""
    # Mock database and collections
    with mock.patch('SVD.DB') as MockDB:
        # Setup mock DB
        mock_db = mock.Mock()
        mock_db.movie_db = mock.Mock()
        MockDB.return_value = mock_db
        
        # Setup user_watch_data collection with varying watch times
        watch_data_with_varying_times = [
            {
                "user_id": "user1",
                "movie_id": "1",
                "minute_mpg": "30",  # Short watch time
                "timestamp": datetime.datetime(2023, 1, 5, 10, 0, 0)
            },
            {
                "user_id": "user2",
                "movie_id": "2",
                "minute_mpg": "120",  # Medium watch time
                "timestamp": datetime.datetime(2023, 1, 6, 10, 0, 0)
            },
            {
                "user_id": "user3",
                "movie_id": "3",
                "minute_mpg": "240",  # Long watch time
                "timestamp": datetime.datetime(2023, 1, 7, 10, 0, 0)
            }
        ]
        mock_db.movie_db["user_watch_data"].find.return_value.limit.return_value = watch_data_with_varying_times
        
        # Create SVDPipeline with mocked data
        pipeline = SVDPipeline()
        pipeline.DB = mock_db
        
        # Clean watch data
        clean_watch_df = pipeline.clean_watch_data(mock_db.movie_db)
        
        # Verify ratings were properly normalized between 1 and 5
        assert all(1 <= rating <= 5 for rating in clean_watch_df['rating'].values)
        
        # Verify that longer watch time corresponds to higher rating
        sorted_df = clean_watch_df.sort_values(by='watch_time')
        assert sorted_df.iloc[0]['rating'] < sorted_df.iloc[-1]['rating']

def test_movie_recommendations_order(movie_data):
    """Test that recommendations are returned in order of predicted rating."""
    # Mock database and model
    with mock.patch('SVD.DB') as MockDB:
        with mock.patch('SVD.SVDPipeline.combined_ratings_df', 
                      new_callable=mock.PropertyMock) as mock_combined:
            with mock.patch('SVD.SVDPipeline.movies_df',
                          new_callable=mock.PropertyMock) as mock_movies:
                # Mock data
                mock_combined.return_value = pd.DataFrame({
                    'user_id': ['user1', 'user1', 'user2'],
                    'movie_id': ['1', '2', '3'],
                    'rating': [4.5, 3.0, 4.0]
                })
                
                movie_df = pd.DataFrame(movie_data)
                movie_df['movie_id'] = movie_df['movie_id'].astype(str)
                mock_movies.return_value = movie_df
                
                # Create mock SVD model with deterministic predictions
                mock_model = mock.Mock()
                
                # Create prediction objects with decreasing estimated ratings
                predictions = [
                    mock.Mock(est=4.8, iid='4'),  # Higher prediction
                    mock.Mock(est=4.2, iid='5'),  # Middle prediction
                    mock.Mock(est=3.5, iid='3')   # Lower prediction
                ]
                mock_model.predict.side_effect = predictions
                
                # Create pipeline with mocked data and model
                pipeline = SVDPipeline()
                pipeline.combined_ratings_df = mock_combined.return_value
                pipeline.movies_df = mock_movies.return_value
                pipeline.svd_model = mock_model
                
                # Get recommendations
                recommendations = pipeline.get_recommendations('user1', num_recommendations=3)
                
                # Verify recommendations are in order of decreasing predicted rating
                assert recommendations == ['4', '5', '3']

def test_fetch_collection_data_days_filter():
    """Test that days_back parameter correctly filters data by date."""
    # Mock database and collection
    with mock.patch('SVD.DB') as MockDB:
        # Setup mock DB
        mock_db = mock.Mock()
        mock_db.movie_db = mock.Mock()
        MockDB.return_value = mock_db
        
        # Create test data with different timestamps
        now = datetime.datetime.now()
        test_data = [
            {"user_id": "user1", "movie_id": "1", "timestamp": now - datetime.timedelta(days=10)},
            {"user_id": "user1", "movie_id": "2", "timestamp": now - datetime.timedelta(days=5)},
            {"user_id": "user1", "movie_id": "3", "timestamp": now - datetime.timedelta(days=1)}
        ]
        
        # Mock collection.find to capture the query
        find_mock = mock.Mock()
        find_mock.limit.return_value = test_data
        mock_db.movie_db["test_collection"].find = find_mock
        
        # Create SVDPipeline with mocked data
        pipeline = SVDPipeline()
        pipeline.DB = mock_db
        
        # Test with 7 days filter
        pipeline.fetch_collection_data(mock_db.movie_db, "test_collection", days_back=7)
        
        # Verify the query has the correct timestamp filter
        query_arg = find_mock.call_args[0][0]
        assert "timestamp" in query_arg
        assert "$gte" in query_arg["timestamp"]
        
        # Verify the timestamp cutoff is roughly 7 days ago
        cutoff_timestamp = query_arg["timestamp"]["$gte"]
        days_diff = (now - cutoff_timestamp).days
        assert 6 <= days_diff <= 7  # Allow for small timing differences

def test_genre_extraction():
    """Test that movie genres are correctly extracted from different formats."""
    # Mock database and collections
    with mock.patch('SVD.DB') as MockDB:
        # Setup mock DB
        mock_db = mock.Mock()
        mock_db.movie_db = mock.Mock()
        MockDB.return_value = mock_db
        
        # Setup movie_info collection with different genre formats
        movies_with_different_genres = [
            {
                "movie_id": "1",
                "genres": [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]  # List of dicts
            },
            {
                "movie_id": "2",
                "genres": '[{"id": 35, "name": "Comedy"}]'  # String containing JSON
            },
            {
                "movie_id": "3",
                "genres": "Drama, Romance"  # Simple string
            },
            {
                "movie_id": "4",
                "genres": []  # Empty list
            }
        ]
        mock_db.movie_db["movie_info"].find.return_value.limit.return_value = movies_with_different_genres
        
        # Create SVDPipeline with mocked data
        pipeline = SVDPipeline()
        pipeline.DB = mock_db
        
        # Clean movie data
        clean_movie_df = pipeline.clean_movie_data(mock_db.movie_db)
        
        # Verify genres were correctly extracted
        assert "Action, Adventure" in clean_movie_df['genres'].values
        assert "Comedy" in clean_movie_df['genres'].values
        assert "" in clean_movie_df['genres'].values  # For the empty list and invalid string

def test_data_processing_with_missing_values():
    """Test handling of missing values in dataset."""
    # Mock database and collections
    with mock.patch('SVD.DB') as MockDB:
        # Setup mock DB
        mock_db = mock.Mock()
        mock_db.movie_db = mock.Mock()
        mock_db.user_db = mock.Mock()
        MockDB.return_value = mock_db
        
        # Setup collections with missing values
        movies_with_missing = [
            {"movie_id": "1", "adult": False, "genres": [], "release_date": None, "original_language": "en"},
            {"movie_id": "2", "adult": False, "genres": [{"id": 35, "name": "Comedy"}], "original_language": "en"}  # Missing release_date
        ]
        mock_db.movie_db["movie_info"].find.return_value.limit.return_value = movies_with_missing
        
        ratings_with_missing = [
            {"user_id": "user1", "movie_id": "1", "timestamp": datetime.datetime.now()},  # Missing score
            {"user_id": "user1", "movie_id": "2", "score": 3.5, "timestamp": datetime.datetime.now()}
        ]
        mock_db.movie_db["user_rate_data"].find.return_value.limit.return_value = ratings_with_missing
        
        watch_with_missing = [
            {"user_id": "user2", "movie_id": "1", "timestamp": datetime.datetime.now()},  # Missing minute_mpg
            {"user_id": "user2", "movie_id": "2", "minute_mpg": "120", "timestamp": datetime.datetime.now()}
        ]
        mock_db.movie_db["user_watch_data"].find.return_value.limit.return_value = watch_with_missing
        
        users_with_missing = [
            {"user_id": "user1"},  # Missing demographic info
            {"user_id": "user2", "age": 25, "occupation": "Student", "gender": "F"}
        ]
        mock_db.user_db["user_info"].find.return_value.limit.return_value = users_with_missing
        
        # Create SVDPipeline with mocked data
        with mock.patch('SVD.SVD'):
            with mock.patch('SVD.Dataset'):
                with mock.patch('SVD.train_test_split'):
                    with mock.patch('SVD.Reader'):
                        # This should not raise exceptions
                        pipeline = SVDPipeline()
                        pipeline.load_clean_data()
                        
                        # Verify data with missing values was handled
                        assert len(pipeline.movies_df) == 2
                        assert len(pipeline.ratings_df) == 1  # Only one rating had score
                        
                        # Verify watch data had missing values handled
                        assert len(pipeline.watch_df) == 1  # Only one watch had minute_mpg

def test_handle_malformed_genres():
    """Test handling of malformed genre data."""
    # Mock database and collections
    with mock.patch('SVD.DB') as MockDB:
        # Setup mock DB
        mock_db = mock.Mock()
        mock_db.movie_db = mock.Mock()
        MockDB.return_value = mock_db
        
        # Setup movie_info collection with malformed genres
        movies_with_malformed_genres = [
            {
                "movie_id": "1",
                "genres": "Not a valid JSON string or list"
            },
            {
                "movie_id": "2",
                "genres": [{"not_id": 35, "not_name": "Comedy"}]  # Missing expected keys
            },
            {
                "movie_id": "3",
                "genres": 123  # Not a string or list
            },
            {
                "movie_id": "4",
                "genres": None  # None value
            }
        ]
        mock_db.movie_db["movie_info"].find.return_value.limit.return_value = movies_with_malformed_genres
        
        # Create SVDPipeline with mocked data
        pipeline = SVDPipeline()
        pipeline.DB = mock_db
        
        # This should not raise exceptions
        clean_movie_df = pipeline.clean_movie_data(mock_db.movie_db)
        
        # Verify all rows were processed despite malformed genres
        assert len(clean_movie_df) == 4
        
        # Verify all genre values are strings (empty for malformed data)
        assert all(isinstance(genre, str) for genre in clean_movie_df['genres'].values)
        assert "" in clean_movie_df['genres'].values  # Empty string for malformed genres

def test_empty_database_handling():
    """Test handling of empty database collections."""
    # Mock database and collections
    with mock.patch('SVD.DB') as MockDB:
        # Setup mock DB with empty collections
        mock_db = mock.Mock()
        mock_db.movie_db = mock.Mock()
        mock_db.user_db = mock.Mock()
        MockDB.return_value = mock_db
        
        mock_db.movie_db["movie_info"].find.return_value.limit.return_value = []
        mock_db.movie_db["user_rate_data"].find.return_value.limit.return_value = []
        mock_db.movie_db["user_watch_data"].find.return_value.limit.return_value = []
        mock_db.user_db["user_info"].find.return_value.limit.return_value = []
        
        # Create SVDPipeline with mocked empty data
        with mock.patch('SVD.SVD'):
            with mock.patch('SVD.Dataset'):
                with mock.patch('SVD.train_test_split'):
                    with mock.patch('SVD.Reader'):
                        # This should not raise exceptions
                        pipeline = SVDPipeline()
                        pipeline.load_clean_data()
                        
                        # Verify empty dataframes were created
                        assert len(pipeline.movies_df) == 0
                        assert len(pipeline.ratings_df) == 0
                        assert len(pipeline.watch_df) == 0
                        assert len(pipeline.users_df) == 0
                        
                        # Test recommendations with empty data
                        recommendations = pipeline.get_recommendations('user1')
                        assert recommendations == []

def test_mongodb_connection_error():
    """Test handling of MongoDB connection errors."""
    # Mock MongoClient to raise exception
    with mock.patch('SVD.MongoClient', side_effect=Exception("Connection refused")):
        # This should not raise an exception to the caller
        with pytest.raises(Exception) as exc_info:
            db = DB()
        
        assert "Connection refused" in str(exc_info.value)

def test_prediction_for_user_without_ratings():
    """Test recommendations for a user without any ratings."""
    # Mock database and model
    with mock.patch('SVD.DB') as MockDB:
        with mock.patch('SVD.SVDPipeline.combined_ratings_df', 
                      new_callable=mock.PropertyMock) as mock_combined:
            with mock.patch('SVD.SVDPipeline.movies_df',
                          new_callable=mock.PropertyMock) as mock_movies:
                # Mock data with existing users but not the target user
                mock_combined.return_value = pd.DataFrame({
                    'user_id': ['user1', 'user1', 'user2'],
                    'movie_id': ['1', '2', '3'],
                    'rating': [4.5, 3.0, 4.0]
                })
                
                mock_movies.return_value = pd.DataFrame({
                    'movie_id': ['1', '2', '3', '4', '5'],
                    'genres': ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
                })
                
                # Mock numpy random choice to return a consistent user
                with mock.patch('numpy.random.choice') as mock_choice:
                    mock_choice.return_value = 'user1'
                    
                    # Create mock SVD model
                    mock_model = mock.Mock()
                    # Setup prediction objects
                    pred1 = mock.Mock(est=4.8, iid='4')
                    pred2 = mock.Mock(est=3.5, iid='5')
                    mock_model.predict.side_effect = [pred1, pred2]
                    
                    # Create pipeline with mocked data
                    pipeline = SVDPipeline()
                    pipeline.combined_ratings_df = mock_combined.return_value
                    pipeline.movies_df = mock_movies.return_value
                    pipeline.svd_model = mock_model
                    
                    # Get recommendations for new user
                    recommendations = pipeline.get_recommendations('new_user')
                    
                    # Verify a random user was chosen and recommendations returned
                    mock_choice.assert_called_once()
                    assert len(recommendations) > 0

def test_model_fallback_with_exception():
    """Test model fallback when an exception occurs during prediction."""
    # Mock database and model
    with mock.patch('SVD.DB') as MockDB:
        with mock.patch('SVD.SVDPipeline.combined_ratings_df', 
                      new_callable=mock.PropertyMock) as mock_combined:
            with mock.patch('SVD.SVDPipeline.movies_df',
                          new_callable=mock.PropertyMock) as mock_movies:
                # Mock data
                mock_combined.return_value = pd.DataFrame({
                    'user_id': ['user1', 'user1', 'user2'],
                    'movie_id': ['1', '2', '3'],
                    'rating': [4.5, 3.0, 4.0]
                })
                
                mock_movies.return_value = pd.DataFrame({
                    'movie_id': ['1', '2', '3', '4', '5'],
                    'genres': ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
                })
                
                # Create mock SVD model that raises exception
                mock_model = mock.Mock()
                mock_model.predict.side_effect = Exception("Model prediction error")
                
                # Create pipeline with mocked data and failing model
                pipeline = SVDPipeline()
                pipeline.combined_ratings_df = mock_combined.return_value
                pipeline.movies_df = mock_movies.return_value
                pipeline.svd_model = mock_model
                
                # Mock sample method to return predictable results
                with mock.patch.object(pd.DataFrame, 'sample') as mock_sample:
                    mock_sample.return_value = pd.DataFrame({
                        'movie_id': ['4', '5'],
                        'genres': ['Horror', 'Sci-Fi']
                    })
                    
                    # Get recommendations - should fall back to random sampling
                    recommendations = pipeline.get_recommendations('user1')
                    
                    # Verify model tried to predict but fell back to sample
                    mock_model.predict.assert_called()
                    mock_sample.assert_called_once()
                    assert recommendations == ['4', '5']