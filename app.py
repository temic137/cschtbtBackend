from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from config import Config
from extensions import db, mongo_client
from routes.auth import auth_bp
from routes.chatbot import chatbot_bp
from routes.inventory import inventory_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    CORS(app, supports_credentials=True)
    db.init_app(app)
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(inventory_bp)
    
    with app.app_context():
        db.create_all()
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)