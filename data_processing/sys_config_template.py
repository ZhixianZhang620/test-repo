import os

DATA_DIR = './Kafka_data'
recommend_log_path = 'recommend_log.csv'
USER_RECOMMEND_LOG = os.path.join(DATA_DIR, recommend_log_path)

IP = '128.2.204.215'
USER_API_URL = 'http://'+IP+':8080/user'
MOVIE_API_URL = 'http://'+IP+':8080/movie'

# MongoDB Configuration
MONGO_URI = "" # replace with your mongo URI
USER_DB = 'user_database'
MOVIE_DB = 'movie_database'
LOG_DB = 'log_database'
USER_RATE_COLLECTION = 'user_rate_data'
USER_WATCH_COLLECTION = 'user_watch_data'
USER_DATA_COLLECTION = 'user_info'
MOVIE_DATA_COLLECTION = 'movie_info'
RECOMMENDATION_LOG = 'recommendation_log'