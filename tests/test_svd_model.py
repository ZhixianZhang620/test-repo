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

@pytest.fixture
def mock_db():
    """Create a mock database for testing."""
    db_mock = mock.Mock()
    db_mock.movie_db = mock.Mock()
    db_mock.user_db = mock.Mock()
    
    # Setup mock collections
    db_mock.movie_db["movie_info"] = mock.Mock()
    db_mock.movie_db["user_rate_data"] = mock.Mock()
    db_mock.movie_db["user_watch_data"] = mock.Mock()
    db_mock.user_db["user_info"] = mock.Mock()
    
    return db_mock

@pytest.fixture
def sample_movie_data():
    """Create sample movie data for testing."""
    return [{
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
    }]

@pytest.fixture
def sample_user_data():
    """Create sample user data for testing."""
    return [{
        'user_id': 'user1',
        'age': 30,
        'occupation': 'Engineer',
        'gender': 'M'
    },
    {
        'user_id': 'user2',
        'age': 25,
        'occupation': 'Student',
        'gender': 'F'
    }]

@pytest.fixture
def sample_ratings_data():
    """Create sample ratings data for testing."""
    return [{
        'user_id': 'user1',
        'movie_id': '1',
        'score': 4.5,
        'timestamp': datetime.datetime.now()
    },
    {
        'user_id': 'user1',
        'movie_id': '2',
        'score': 3.0,
        'timestamp': datetime.datetime.now()
    },
    {
        'user_id': 'user2',
        'movie_id': '1',
        'score': 5.0,
        'timestamp': datetime.datetime.now()
    }]

@pytest.fixture
def sample_watch_data():
    """Create sample watch data for testing."""
    return [{
        'user_id': 'user1',
        'movie_id': '1',
        'minute_mpg': '120',
        'timestamp': datetime.datetime.now()
    },
    {
        'user_id': 'user2',
        'movie_id': '2',
        'minute_mpg': '90',
        'timestamp': datetime.datetime.now()
    }]

def test_db_connection():
    """Test that DB class can be initialized."""
    with mock.patch('SVD.MongoClient') as mock_client:
        db = DB()
        mock_client.assert_called_once()
        assert hasattr(db, 'user_db')
        assert hasattr(db, 'movie_db')

def test_fetch_collection_data(mock_db, sample_movie_data):
    """Test fetching data from a collection."""
    mock_db.movie_db["movie_info"].find.return_value.limit.return_value = sample_movie_data
    
    with mock.patch('SVD.DB', return_value=mock_db):
        svd_pipeline = SVDPipeline()
        svd_pipeline.DB = mock_db
        
        result = svd_pipeline.fetch_collection_data(mock_db.movie_db, "movie_info")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_movie_data)
        
        # Test with days_back filter
        svd_pipeline.fetch_collection_data(mock_db.movie_db, "movie_info", days_back=7)
        # Verify query with filter was used
        args, _ = mock_db.movie_db["movie_info"].find.call_args
        assert "$gte" in args[0]["timestamp"]

def test_clean_movie_data(mock_db, sample_movie_data):
    """Test cleaning movie data."""
    mock_db.movie_db["movie_info"].find.return_value.limit.return_value = sample_movie_data
    
    with mock.patch('SVD.DB', return_value=mock_db):
        svd_pipeline = SVDPipeline()
        svd_pipeline.DB = mock_db
        
        result = svd_pipeline.clean_movie_data(mock_db.movie_db)
        assert 'movie_id' in result.columns
        assert 'genres' in result.columns
        assert 'Action, Adventure' in result['genres'].values
        assert 'Comedy' in result['genres'].values

def test_clean_user_data(mock_db, sample_user_data):
    """Test cleaning user data."""
    mock_db.user_db["user_info"].find.return_value.limit.return_value = sample_user_data
    
    with mock.patch('SVD.DB', return_value=mock_db):
        svd_pipeline = SVDPipeline()
        svd_pipeline.DB = mock_db
        
        result = svd_pipeline.clean_user_data(mock_db.user_db)
        assert 'age' in result.columns
        assert 'occupation' in result.columns
        assert 'gender' in result.columns
        assert 30 in result['age'].values
        assert 'Engineer' in result['occupation'].values

def test_clean_ratings_data(mock_db, sample_ratings_data):
    """Test cleaning ratings data."""
    mock_db.movie_db["user_rate_data"].find.return_value.limit.return_value = sample_ratings_data
    
    with mock.patch('SVD.DB', return_value=mock_db):
        svd_pipeline = SVDPipeline()
        svd_pipeline.DB = mock_db
        
        result = svd_pipeline.clean_ratings_data(mock_db.movie_db)
        assert 'user_id' in result.columns
        assert 'movie_id' in result.columns
        assert 'score' in result.columns
        assert '1' in result['movie_id'].values
        assert 4.5 in result['score'].values

def test_clean_watch_data(mock_db, sample_watch_data):
    """Test cleaning watch data."""
    mock_db.movie_db["user_watch_data"].find.return_value.limit.return_value = sample_watch_data
    
    with mock.patch('SVD.DB', return_value=mock_db):
        svd_pipeline = SVDPipeline()
        svd_pipeline.DB = mock_db
        
        result = svd_pipeline.clean_watch_data(mock_db.movie_db)
        assert 'user_id' in result.columns
        assert 'movie_id' in result.columns
        assert 'rating' in result.columns
        assert '1' in result['movie_id'].values
        assert all(1 <= rating <= 5 for rating in result['rating'].values)

def test_load_clean_data():
    """Test loading and cleaning all data types."""
    with mock.patch('SVD.SVDPipeline.clean_movie_data') as mock_clean_movie:
        with mock.patch('SVD.SVDPipeline.clean_user_data') as mock_clean_user:
            with mock.patch('SVD.SVDPipeline.clean_ratings_data') as mock_clean_ratings:
                with mock.patch('SVD.SVDPipeline.clean_watch_data') as mock_clean_watch:
                    # Mock return values
                    mock_clean_movie.return_value = pd.DataFrame({'movie_id': ['1', '2']})
                    mock_clean_user.return_value = pd.DataFrame({'user_id': ['user1', 'user2']})
                    mock_clean_ratings.return_value = pd.DataFrame({'user_id': ['user1'], 'movie_id': ['1'], 'score': [4.5]})
                    mock_clean_watch.return_value = pd.DataFrame({'user_id': ['user2'], 'movie_id': ['2'], 'rating': [3.0]})
                    
                    with mock.patch('SVD.DB'):
                        svd_pipeline = SVDPipeline()
                        # Override the model creation in __init__
                        svd_pipeline.svd_model = mock.Mock()
                        
                        svd_pipeline.load_clean_data()
                        
                        # Verify all clean methods were called
                        mock_clean_movie.assert_called_once()
                        mock_clean_user.assert_called_once()
                        mock_clean_ratings.assert_called_once()
                        mock_clean_watch.assert_called_once()
                        
                        # Verify dataframes were created
                        assert hasattr(svd_pipeline, 'movies_df')
                        assert hasattr(svd_pipeline, 'users_df')
                        assert hasattr(svd_pipeline, 'ratings_df')
                        assert hasattr(svd_pipeline, 'watch_df')

@mock.patch('SVD.SVD')
@mock.patch('SVD.Dataset')
@mock.patch('SVD.train_test_split')
def test_train_and_save_model(mock_split, mock_dataset, mock_svd):
    """Test the model training process."""
    # Create mock objects
    mock_trainset = mock.Mock()
    mock_testset = mock.Mock()
    mock_split.return_value = (mock_trainset, mock_testset)
    mock_model = mock.Mock()
    mock_svd.return_value = mock_model
    mock_data = mock.Mock()
    mock_dataset.load_from_df.return_value = mock_data
    
    with mock.patch('SVD.SVDPipeline.load_clean_data'):
        with mock.patch.object(SVDPipeline, 'combined_ratings_df', 
                              new_callable=mock.PropertyMock) as mock_combined:
            mock_combined.return_value = pd.DataFrame({
                'user_id': ['user1', 'user2'],
                'movie_id': ['1', '2'],
                'rating': [4.5, 3.0]
            })
            
            with mock.patch('SVD.DB'):
                # Create pipeline with a mocked combined_ratings_df
                svd_pipeline = SVDPipeline()
                svd_pipeline.combined_ratings_df = mock_combined.return_value
                
                # Override the model creation in __init__
                svd_pipeline.svd_model = mock.Mock()
                
                # Call method under test
                model = svd_pipeline.train_and_save_model()
                
                # Verify data was loaded and model was trained
                mock_dataset.load_from_df.assert_called_once()
                mock_split.assert_called_once()
                mock_svd.assert_called_once()
                mock_model.fit.assert_called_once_with(mock_trainset)
                assert model == mock_model

def test_get_recommendations_existing_user():
    """Test getting recommendations for an existing user."""
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
                'genres': ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror']
            })
            
            # Create mock SVD model
            mock_model = mock.Mock()
            # Setup prediction objects
            pred1 = mock.Mock(est=4.8, iid='4')
            pred2 = mock.Mock(est=3.5, iid='5')
            mock_model.predict.side_effect = [pred1, pred2]
            
            with mock.patch('SVD.DB'):
                # Create pipeline with mocked data and model
                svd_pipeline = SVDPipeline()
                svd_pipeline.combined_ratings_df = mock_combined.return_value
                svd_pipeline.movies_df = mock_movies.return_value
                svd_pipeline.svd_model = mock_model
                
                # Get recommendations
                recommendations = svd_pipeline.get_recommendations('user1', num_recommendations=2)
                
                # Verify predictions were made for unwatched movies
                assert mock_model.predict.call_count == 2
                # Verify the results include the top predicted movies
                assert recommendations == ['4', '5']

def test_get_recommendations_new_user():
    """Test getting recommendations for a new user."""
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
                'genres': ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror']
            })
            
            # Mock numpy random choice
            with mock.patch('numpy.random.choice') as mock_choice:
                mock_choice.return_value = 'user1'
                
                # Create mock SVD model
                mock_model = mock.Mock()
                # Setup prediction objects
                pred1 = mock.Mock(est=4.8, iid='4')
                pred2 = mock.Mock(est=3.5, iid='5')
                mock_model.predict.side_effect = [pred1, pred2]
                
                with mock.patch('SVD.DB'):
                    # Create pipeline with mocked data and model
                    svd_pipeline = SVDPipeline()
                    svd_pipeline.combined_ratings_df = mock_combined.return_value
                    svd_pipeline.movies_df = mock_movies.return_value
                    svd_pipeline.svd_model = mock_model
                    
                    # Get recommendations for non-existent user
                    recommendations = svd_pipeline.get_recommendations('new_user', num_recommendations=2)
                    
                    # Verify a random user was chosen
                    mock_choice.assert_called_once()
                    # Verify predictions were made
                    assert mock_model.predict.call_count == 2
                    # Verify results
                    assert recommendations == ['4', '5']

def test_get_recommendations_no_unwatched_movies():
    """Test recommendations when user has watched all available movies."""
    with mock.patch('SVD.SVDPipeline.combined_ratings_df', 
                   new_callable=mock.PropertyMock) as mock_combined:
        with mock.patch('SVD.SVDPipeline.movies_df',
                        new_callable=mock.PropertyMock) as mock_movies:
            # Mock data where user has watched all movies
            mock_combined.return_value = pd.DataFrame({
                'user_id': ['user1', 'user1', 'user1'],
                'movie_id': ['1', '2', '3'],
                'rating': [4.5, 3.0, 4.0]
            })
            
            mock_movies.return_value = pd.DataFrame({
                'movie_id': ['1', '2', '3'],
                'genres': ['Action', 'Comedy', 'Drama']
            })
            
            with mock.patch('SVD.DB'):
                # Create pipeline with mocked data
                svd_pipeline = SVDPipeline()
                svd_pipeline.combined_ratings_df = mock_combined.return_value
                svd_pipeline.movies_df = mock_movies.return_value
                svd_pipeline.svd_model = mock.Mock()
                
                # Get recommendations when all movies have been watched
                with mock.patch.object(pd.DataFrame, 'sample') as mock_sample:
                    mock_sample.return_value = pd.DataFrame({
                        'movie_id': ['2', '3'],
                        'genres': ['Comedy', 'Drama']
                    })
                    
                    recommendations = svd_pipeline.get_recommendations('user1', num_recommendations=2)
                    
                    # Verify sample was called
                    mock_sample.assert_called_once()
                    # Verify random recommendations were returned
                    assert recommendations == ['2', '3']

def test_handle_empty_dataframes():
    """Test handling empty dataframes."""
    with mock.patch('SVD.SVDPipeline.combined_ratings_df', 
                   new_callable=mock.PropertyMock) as mock_combined:
        with mock.patch('SVD.SVDPipeline.movies_df',
                        new_callable=mock.PropertyMock) as mock_movies:
            # Mock empty dataframes
            mock_combined.return_value = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
            mock_movies.return_value = pd.DataFrame(columns=['movie_id', 'genres'])
            
            with mock.patch('SVD.DB'):
                # Create pipeline with empty data
                svd_pipeline = SVDPipeline()
                svd_pipeline.combined_ratings_df = mock_combined.return_value
                svd_pipeline.movies_df = mock_movies.return_value
                svd_pipeline.svd_model = mock.Mock()
                
                # This should not raise exceptions
                recommendations = svd_pipeline.get_recommendations('user1')
                
                # With no data, we expect an empty list or similar
                assert recommendations == []