from flask import Flask, jsonify, make_response
from SVD import SVDPipeline

app = Flask(__name__)
# Initialize the SVDPipeline globally - this avoids reloading the model for each request
svd_pipeline = SVDPipeline()

@app.route("/")
def home():
    return "ML Service is Running!"

@app.route("/recommend/<int:userid>")
def recommend(userid):
    # TODO call SVD Model here and return result
    try:
        recommended_movies = svd_pipeline.get_recommendations(userid)
        print(f"{recommended_movies}")
        response_obj = make_response(recommended_movies)
        response_obj.status_code = 200
        return response_obj
    except Exception as e:
        fail_response_obj = make_response(
            {"error": f"Request failed: {str(e)}"}
        )
        fail_response_obj.status_code = 500    
        return fail_response_obj

if __name__ == "__main__":
    # Load config from config.py
    app.config.from_object("config.Config")
    
    host = app.config.get("HOST", "128.2.205.110")
    port = app.config.get("PORT", 8083)
    debug = app.config.get("DEBUG", False)
    app.run(host, port, debug)