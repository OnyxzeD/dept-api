import json
from json import JSONEncoder

from flask import jsonify, make_response


def ok(data, message):
    res = {"data": data, "message": message}

    return make_response(jsonify(res)), 200


def badRequest(data, message):
    res = {"data": data, "message": message}

    return make_response(jsonify(res)), 500
	
	
def customRequest(data, message, status):
    res = {"data": data, "message": message, "status": status}

    return make_response(jsonify(res)), status


class Encoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
