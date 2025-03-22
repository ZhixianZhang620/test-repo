from flask import Flask, jsonify
import pandas as pd
import numpy as np
import requests
from flask import make_response

app = Flask(__name__)
ml_host = app.config.get("ML_HOST", "128.2.205.110")
ml_port = app.config.get("ML_PORT", 8083)

@app.route("/")
def home():
    return "Movie Recommendation Service is Running!"

@app.route("/recommend/<int:userid>")
def recommend(userid):
    '''
        Process request with parameter <userid> and respond with an ordered 
        comma-separated list of up to 20 movie IDs in a single line.
    '''
    ml_url = f'http://{ml_host}:{ml_port}/recommend/{userid}'

    try:
        response = requests.get(ml_url, timeout=5)
        if response.status_code == 200:
            movie_list = response.json()
            movie_string = ",".join(str(movie) for movie in movie_list)
            print(f"success response {movie_string}")
            
            response_obj = make_response(movie_string)
            response_obj.status_code = 200
            return response_obj
        else:
            fail_response_obj = make_response(
                {"error": f"ML service error: {response.status_code}"}
            )
            fail_response_obj.status_code = response.status_code
            return fail_response_obj
    except requests.exceptions.Timeout:
        print("timeout!")
        fail_response_obj = make_response(
            {"error": "ML service timed out"}
        )
        fail_response_obj.status_code = 504
        return fail_response_obj
    except requests.exceptions.RequestException as e:
        print(f"request failed {str(e)}")
        fail_response_obj = make_response(
            {"error": f"Request failed: {str(e)}"}
        )
        fail_response_obj.status_code = 500
        return fail_response_obj
    
if __name__ == "__main__":
    # Load config from config.py
    app.config.from_object("config.Config")
    
    host = app.config.get("HOST", "128.2.205.110")
    port = app.config.get("PORT", 8082)
    debug = app.config.get("DEBUG", False)
    app.run(host, port, debug)