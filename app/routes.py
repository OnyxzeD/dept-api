from flask import jsonify, request
from flask_cors import CORS
from flask_login import logout_user

from app import app
from app.controllers import UserController, TweetController, DetectionController

# enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
@app.route("/index")
def index():
    return "DEpressive Personality Test - API Service"


@app.route("/anonym", methods=["GET"])
def anonym():
    return UserController.createAnonym()


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify("pong!")


@app.route("/users", methods=["POST", "GET"])
def users():
    if request.method == "GET":
        return UserController.index()
    else:
        return UserController.store()


@app.route("/users/<id>", methods=["PUT", "GET", "DELETE"])
def usersDetail(id):
    if request.method == "GET":
        return UserController.show(id)
    elif request.method == "PUT":
        return UserController.update(id)
    elif request.method == "DELETE":
        return UserController.delete(id)


@app.route("/login", methods=["POST"])
def login():
    return UserController.login()


@app.route("/logout")
def logout():
    logout_user()
    return "Successfully logged out!"


@app.route("/hash", methods=["GET"])
def hash():
    return UserController.hash()


@app.route("/tweets", methods=["POST", "GET"])
def tweets():
    if request.method == "GET":
        return TweetController.index()
    else:
        return TweetController.store()


@app.route("/tweets/<id>", methods=["PUT", "GET", "DELETE"])
def tweetsDetail(id):
    if request.method == "GET":
        return TweetController.show(id)
    elif request.method == "PUT":
        return TweetController.update(id)
    elif request.method == "DELETE":
        return TweetController.delete(id)


# @app.route("/scrape", methods=["POST"])
# def scrape():
#     return DetectionController.scrape()


@app.route("/detection", methods=["POST", "GET"])
def detection():
    if request.method == "GET":
        return DetectionController.index()
    else:
        return DetectionController.store()


@app.route("/detection/<detection_id>", methods=["PUT", "GET", "DELETE"])
def detectionDetail(detection_id):
    if request.method == "GET":
        return DetectionController.show(detection_id)
    elif request.method == "PUT":
        return DetectionController.update(detection_id)
    elif request.method == "DELETE":
        return DetectionController.delete(detection_id)


@app.route("/detection-log/<user_id>", methods=["GET"])
def detectionByUser(user_id):
    if request.method == "GET":
        return DetectionController.byUser(user_id)


@app.route("/dashboard-data", methods=["GET"])
def dashboard():
    return DetectionController.dashboard()


@app.route("/classify", methods=["POST"])
def classify():
    return DetectionController.classify()


@app.route("/train", methods=["GET"])
def train():
    return DetectionController.train()


@app.route("/tester", methods=["GET"])
def tester():
    return DetectionController.tester()


@app.route("/tester2", methods=["GET"])
def tester2():
    return DetectionController.tester2()
