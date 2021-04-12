import random

from app import app, db, response
from app.models import Users
from flask import jsonify, request
from flask_login import current_user, login_user


def index():
    try:
        users = Users.query.all()
        data = transform(users)
        return response.ok(data, "User Lists")
    except Exception as e:
        return response.badRequest([], str(e))


def show(userId):
    try:
        users = Users.query.filter_by(id=userId).first()
        if not users:
            return response.badRequest([], "User not found")

        data = singleTransform(users)
        return response.ok(data, "Users Detail")
    except Exception as e:
        return response.badRequest([], str(e))


def getAvatar(gender):
    females = [
        "female-enfj-protagonist.svg",
        "female-enfj-protagonist-p-s1-v1.svg",
        "female-enfj-protagonist-s1.svg",
        "female-entj-commander-c-s3-v1.svg",
        "female-entj-commander-p-s3-v1.svg",
        "female-entj-commander-s3.svg",
        "female-infp-mediator-c-s3-v1.svg",
        "female-infp-mediator-p-s3-v1.svg",
        "female-infp-mediator-s3.svg",
        "female-esfj-consul-c-s3-v1.svg",
        "female-esfj-consul-p-s3-v1.svg",
        "female-esfj-consul-s3.svg",
    ]

    males = [
        "male-enfj-protagonist.svg",
        "male-enfj-protagonist-p-s3-v1.svg",
        "male-enfj-protagonist-s3.svg",
        "male-esfp-entertainer-c-s3-v1.svg",
        "male-esfp-entertainer-p-s3-v1.svg",
        "male-esfp-entertainer-s3.svg",
        "male-intp-logician-c-s2-v1.svg",
        "male-intp-logician-p-s2-v1.svg",
        "male-intp-logician-s2.svg",
        "male-esfj-consul-c-s3-v1.svg",
        "male-esfj-consul-p-s3-v1.svg",
        "male-esfj-consul-s3.svg",
    ]

    avatar = random.choice(males)

    if gender == "Female":
        avatar = random.choice(females)

    return avatar


def store():
    try:
        name = request.json["name"]
        email = request.json["email"]
        password = request.json["password"]
        gender = request.json["gender"]
        avatar = getAvatar(gender)

        if request.json["bio"].strip():
            bio = request.json["bio"]
        else:
            bio = "Some meaningless made up words"

        if request.json["role"].strip():
            role = request.json["role"]
        else:
            role = "client"

        user = Users(
            name=name, email=email, gender=gender, avatar=avatar, bio=bio, role=role
        )
        user.setPassword(password)
        db.session.add(user)
        db.session.commit()

        data = {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "gender": user.gender,
            "avatar": user.avatar,
        }
        return response.ok(data, "Successfully create data!")

    except Exception as e:
        return response.badRequest([], str(e))


def update(userId):
    try:
        name = request.json["name"]
        email = request.json["email"]
        password = request.json["password"]
        gender = request.json["gender"]
        bio = request.json["bio"]

        user = Users.query.filter_by(id=userId).first()

        if name.strip():
            user.name = name

        if email.strip():
            user.email = email

        if password.strip():
            user.setPassword(password)

        if gender.strip() != user.gender:
            user.gender = gender
            user.avatar = getAvatar(gender)

        if bio.strip():
            user.bio = bio

        user.setUpdated()
        db.session.commit()

        return response.ok("", "Successfully update data!")

    except Exception as e:
        return response.badRequest([], str(e))


def delete(userId):
    try:
        user = Users.query.filter_by(id=userId).first()
        if not user:
            return response.badRequest([], "User not found")

        db.session.delete(user)
        db.session.commit()

        return response.ok("", "Successfully delete data!")
    except Exception as e:
        return response.badRequest([], str(e))


def createAnonym():
    try:
        total = Users.query.count() + 1
        numStr = str(total)
        name = "User" + numStr.zfill(3)
        email = "anonymous" + numStr.zfill(3) + "@mail.com"
        password = "anonymous"
        gender = "Male"
        avatar = getAvatar(gender)
        bio = "Some meaningless made up words"
        role = "client"

        user = Users(
            name=name, email=email, gender=gender, avatar=avatar, bio=bio, role=role
        )
        user.setPassword(password)
        db.session.add(user)
        db.session.commit()

        data = {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "gender": user.gender,
            "avatar": user.avatar,
        }
        return response.ok(data, "Successfully create data!")

    except Exception as e:
        return response.badRequest([], str(e))


def login():
    try:

        if current_user.is_authenticated:
            return response.ok(current_user.name, "Already Logged in!")

        email = request.json["email"]
        password = request.json["password"]
        remember_me = request.json["remember_me"]

        user = Users.query.filter_by(email=email).first()
        if user is None or not user.checkPassword(password):
            return response.customRequest([], "Invalid email or password", 401)

        login_user(user, remember=remember_me)
        data = {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "gender": user.gender,
            "avatar": user.avatar,
            "bio": user.bio,
            "role": user.role,
            "created_at": user.created_at,
        }
        return response.ok(data, "Successfully logged in!")

    except Exception as ex:
        return response.badRequest([], str(ex))


def hash():
    try:
        user = Users(name="asd", email="asd")
        user.setPassword("secret")

        data = {"hash": user.password}
        return response.ok(data, "Successfully create data!")

    except Exception as ex:
        return response.badRequest([], "Something gone wrong with the sql")


def singleTransform(users):
    data = {
        "id": users.id,
        "name": users.name,
        "email": users.email,
        "gender": users.gender,
        "avatar": users.avatar,
        "bio": users.bio,
        "role": users.role,
        "created_at": users.created_at.strftime("%Y-%m-%d %H:%M"),
    }

    return data


def transform(users):
    array = []
    for u in users:
        array.append(singleTransform(u))
    return array
