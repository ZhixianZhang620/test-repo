from datetime import datetime
import sys
from kafka import KafkaConsumer
import os

from pymongo import MongoClient
import requests
import logging

from data_processing.sys_config import DATA_DIR, LOG_DB, MONGO_URI, MOVIE_API_URL, MOVIE_DATA_COLLECTION, MOVIE_DB, RECOMMENDATION_LOG, USER_API_URL, USER_DATA_COLLECTION, USER_DB, USER_RATE_COLLECTION, USER_RECOMMEND_LOG, USER_WATCH_COLLECTION

# logging any info or error msg by the program
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class DB:
    def __init__(self):
        # mongo DB database and collections setup
        self.client = MongoClient(MONGO_URI)
        self.user_db = self.client[USER_DB]   # user_info
        self.movie_db = self.client[MOVIE_DB] # movie_info, user_watch_data, user_rate_data
        self.log_db = self.client[LOG_DB]
        
        self.user_info = self.user_db[USER_DATA_COLLECTION]
        self.movie_info = self.movie_db[MOVIE_DATA_COLLECTION]
        self.user_rate = self.movie_db[USER_RATE_COLLECTION]
        self.user_watch = self.movie_db[USER_WATCH_COLLECTION]
        self.recommendation_log = self.log_db[RECOMMENDATION_LOG]

# Kafka consumer class
class KafkaConsumerApp:
    def __init__(self, topic, bootstrap_servers):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',  # Start from the latest message
            enable_auto_commit=True,       # Enable auto-commit of offsets
            # How often to tell Kafka, an offset has been read
            auto_commit_interval_ms=1000
        )
        
        self.DB = DB()
        print(f'Kafka Consumer initialized for topic: {topic}')

    def process_user_info(self, user_id):
        try:
            response = requests.get(f'{USER_API_URL}/{user_id}')
            if response.status_code == 200:
                api_data = response.json()
                age = int(api_data["age"])
                occupation = api_data["occupation"]
                gender = api_data["gender"]
                
                # Add both the event time and a collection timestamp (current date)
                collection_date = datetime.now()
                
                user_data = {
                    "timestamp": collection_date,
                    "user_id": user_id,
                    "age": age,
                    "occupation": occupation,
                    "gender": gender
                }
                self.DB.user_info.update_one(
                    {"user_id": user_id},  # Find the user by user_id
                    {"$set": user_data},  # Update the user data if found
                    upsert=True  # If no match is found, insert the document
                )
            else:
                logging.error(f'API fetch failed: {response.status_code}')
        except Exception as e:
            logging.error(f'error inserting user info to db: {e}')
            return
            
    def process_movie_info(self, movie_id):
        try:
            response = requests.get(f'{MOVIE_API_URL}/{movie_id}')
            if response.status_code == 200:
                api_data = response.json()
                adult = api_data["adult"]
                genres = str(api_data["genres"])
                release_date = api_data["release_date"]
                if release_date:
                    release_date = datetime.strptime(api_data["release_date"], "%Y-%m-%d")
                else:
                    release_date = "unknown"
                    
                original_language = api_data["original_language"]
                
                # Add both the event time and a collection timestamp (current date)
                collection_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                movie_data = {
                    "timestamp": collection_date,
                    "movie_id": movie_id,
                    "adult": adult,
                    "genres": genres,
                    "release_date": release_date,
                    "original_language": original_language
                }
                self.DB.movie_info.update_one(
                    {"movie_id": movie_id},  # Find the movie by movie_id
                    {"$set": movie_data},  # Update the movie data if found
                    upsert=True  # If no match is found, insert the document
                )
                
            else:
                print(f'API fetch failed: {response.status_code}')
        except Exception as e:
            logging.error(f'error inserting movie info to db: {e}')
            return

    def process_user_rate(self, rate_msg):
        try:
            time, user_id, other = rate_msg.split(',')
            movie_id, score = other[10:].split('=')
            
            self.process_user_info(user_id)
            self.process_movie_info(movie_id)

            cleaned_time = ''.join(c if c.isnumeric() or c in "-T:" else '' for c in time)
            if len(cleaned_time) == 16:  # 'YYYY-MM-DDTHH:MM' format
                cleaned_time += ":00"  # Automatically add ":00" for seconds
                
            # Add both the event time and a collection timestamp (current date)
            collection_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            user_rate_data = {
                "timestamp": collection_date,
                "time": datetime.strptime(cleaned_time, '%Y-%m-%dT%H:%M:%S'),
                "user_id": user_id,
                "movie_id": movie_id,
                "score": float(score)
            }
            self.DB.user_rate.insert_one(user_rate_data)
        except Exception as e:
            logging.error(f'error inserting to user rate db: {e}')
            return
    
    def process_user_watch_history(self, watch_msg):
        try:
            time, user_id, other = watch_msg.split(',')
            movie_id, minute_mpg = other[12:].split('/')
            
            self.process_user_info(user_id)
            self.process_movie_info(movie_id)
                
            cleaned_time = ''.join(c if c.isnumeric() or c in "-T:" else '' for c in time)
            if len(cleaned_time) == 16:  # 'YYYY-MM-DDTHH:MM' format
                cleaned_time += ":00"  # Automatically add ":00" for seconds
            
            # Add both the event time and a collection timestamp (current date)
            collection_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            user_watch_data = {
                "timestamp": collection_date,
                "time": datetime.strptime(cleaned_time, '%Y-%m-%dT%H:%M:%S'),
                "user_id": user_id,
                "movie_id": movie_id,
                "minute_mpg": minute_mpg
            }
            self.DB.user_watch.insert_one(user_watch_data)
        except Exception as e:
            logging.error(f'error inserting to user watch db: {e}')
            return
    
    def process_recommendation_result(self, recommendation_msg):
        """
            Given recommendation log message, retrieve response process time, user id, request status, and recommendation list result 
        """
        try:
            recommendationList = recommendation_msg.split(',')
            time = recommendationList[0].strip()
            user_id = recommendationList[1].strip()
            status_code = int(recommendationList[3].replace('status', '').strip())
            
            # if response failed, do not record on db
            if status_code != 200:
                logging.error(f'recommendation response failed: {recommendationList}')
                return
            
            recommendation_results = recommendationList[4:24]
            recommendation_results[0] = recommendation_results[0].replace('result:', '').strip()
            response_time = recommendationList[-1].replace('ms', '').strip()

            cleaned_time = ''.join(c if c.isnumeric() or c in "-T:" else '' for c in time)
            if len(cleaned_time) >= 19:
                cleaned_time = cleaned_time[:19]
                
            recommendation_log = {
                "time": datetime.strptime(cleaned_time, '%Y-%m-%dT%H:%M:%S'),
                "user_id": user_id,
                "status_code": status_code,
                "recommendation_results": recommendation_results,
                "response_time": int(response_time)
            }
            
            self.DB.recommendation_log.insert_one(recommendation_log)
        except Exception as e:
            logging.error(f'error inserting to recommendation log db: {e}')
            return

    def consume_messages(self):
        try:
            for message in self.consumer:
                message_str = message.value.decode("utf-8")
                if 'GET /data/m/' in message_str:
                    self.process_user_watch_history(message_str)
                elif 'GET /rate/' in message_str:
                    self.process_user_rate(message_str)
                elif 'recommendation request' in message_str:
                    self.process_recommendation_result(message_str)
        except ConnectionError as e:
            logging.error(f"Kafka connection failed: {e}")
            self.consumer.close()  # Close the Kafka consumer connection
            sys.exit(1)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        
    files_to_remove = [USER_RECOMMEND_LOG]
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
        
    # Replace with your Kafka topic, broker address
    topic = 'movielog8' # our team
    bootstrap_servers = ['localhost:9092']

    consumer_app = KafkaConsumerApp(topic, bootstrap_servers)
    consumer_app.consume_messages()