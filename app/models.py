from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app import db
from app import login


class Users(UserMixin, db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), index=True, unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    gender = db.Column(db.String(12), nullable=False)
    avatar = db.Column(db.String(255))
    bio = db.Column(db.String(255))
    role = db.Column(db.String(15), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now)
    histories = db.relationship("Detection", backref="history", lazy="dynamic")

    def __repr__(self):
        return "<User {}>".format(self.name)

    def setPassword(self, password):
        self.password = generate_password_hash(password)

    def checkPassword(self, password):
        return check_password_hash(self.password, password)

    def setUpdated(self):
        self.updated_at = datetime.now()

    @login.user_loader
    def load_user(self, user_id):
        return Users.query.get(int(user_id))


class Tweets(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    account = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(15), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return "<Tweet {}>".format(self.username)

    def setUpdated(self):
        self.updated_at = datetime.now()


class Detection(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    account = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(255), nullable=False)
    start = db.Column(db.Date)
    end = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now)
    details = db.relationship("History", backref="history", lazy="dynamic")

    def __repr__(self):
        return "<Detection {}>".format(self.account)

    def setUpdated(self):
        self.updated_at = datetime.now()


class History(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection.id"))
    raw_tweet = db.Column(db.Text, nullable=False)
    processed_tweet = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime)
    depressive = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return "<History {}>".format(self.account)

    def setUpdated(self):
        self.updated_at = datetime.now()
