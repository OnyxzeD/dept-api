from datetime import datetime

from app import db, response
from app.models import Tweets
from flask import request


def index():
    try:
        tweets = Tweets.query.all()
        data = transform(tweets)
        return response.ok(data, "Tweet Dataset Lists")

    except Exception as e:
        return response.badRequest([], str(e))


def show(tweetId):
    try:
        tweets = Tweets.query.filter_by(id=tweetId).first()
        if not tweets:
            return response.badRequest([], "Data not found")

        data = singleTransform(tweets)
        return response.ok(data, "Detail")
    except Exception as e:
        return response.badRequest([], str(e))


def store():
    try:
        account = request.json["account"]
        content = request.json["content"]
        label = request.json["label"]

        tweet = Tweets(account=account, content=content, label=label)
        db.session.add(tweet)
        db.session.commit()

        data = {
            "id": tweet.id,
            "account": tweet.account,
            "content": tweet.content,
            "label": tweet.label,
        }
        return response.ok(data, "Successfully create data!")

    except Exception as e:
        return response.badRequest([], str(e))


def update(tweetId):
    try:
        account = request.json["account"]
        content = request.json["content"]
        label = request.json["label"]

        tweet = Tweets.query.filter_by(id=tweetId).first()

        if account.strip():
            tweet.account = account

        if content.strip():
            tweet.content = content

        if label.strip():
            tweet.label = label

        tweet.setUpdated()
        db.session.commit()

        return response.ok("", "Successfully update data!")

    except Exception as e:
        return response.badRequest([], str(e))


def delete(tweetId):
    try:
        tweet = Tweets.query.filter_by(id=tweetId).first()
        if not tweet:
            return response.badRequest([], "Data not found")

        db.session.delete(tweet)
        db.session.commit()

        return response.ok("", "Successfully delete data!")
    except Exception as e:
        return response.badRequest([], str(e))


def singleTransform(tweets):
    data = {
        "id": tweets.id,
        "account": tweets.account,
        "content": tweets.content,
        "label": tweets.label,
    }

    return data


def transform(tweets):
    array = []
    for t in tweets:
        array.append(singleTransform(t))
    return array
