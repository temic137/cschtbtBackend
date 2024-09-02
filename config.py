import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///crm.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'your_secret_key_here'
    SESSION_TYPE = 'filesystem'
    UPLOAD_FOLDER = 'uploads'

