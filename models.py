from extensions import db
from sqlalchemy.dialects.postgresql import JSON  # Import JSON type if using PostgreSQL
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.dialects.postgresql import JSONB

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    chatbots = db.relationship('Chatbot', backref='owner', lazy=True)

class Chatbot(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # data = db.Column(db.Text)
    data = db.Column(JSON)  # JSON column to store both PDF and database data
