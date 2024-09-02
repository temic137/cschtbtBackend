from flask_sqlalchemy import SQLAlchemy
from pymongo import MongoClient

db = SQLAlchemy()
mongo_client = MongoClient('mongodb://localhost:27017/')
mongo_db = mongo_client['your_database_name']
inventory_collection = mongo_db['inventory']