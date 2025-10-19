from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin): 
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    balance = db.Column(db.Numeric, default=0.0)
    
    goals = db.relationship('Goal', backref='user', lazy=True)
    transactions = db.relationship('Transaction', backref='user', lazy=True)

class Goal(db.Model):
    __tablename__ = 'goals'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    goal_type = db.Column(db.String(100))
    cost = db.Column(db.Numeric)
    timeline = db.Column(db.Integer)

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    type = db.Column(db.String(50))  # 'income' or 'expense'
    amount = db.Column(db.Numeric)
    category = db.Column(db.String(100))
    description = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    type = db.Column(db.String(100))
    min_sum = db.Column(db.Numeric)
    max_sum = db.Column(db.Numeric)
    min_term = db.Column(db.Integer)
    max_term = db.Column(db.Integer)
    min_age = db.Column(db.Integer)
    max_age = db.Column(db.Integer)
    description = db.Column(db.Text)
    expected_yield = db.Column(db.String(50))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'min_sum': float(self.min_sum) if self.min_sum else None,
            'max_sum': float(self.max_sum) if self.max_sum else None,
            'min_term': self.min_term,
            'max_term': self.max_term,
            'min_age': self.min_age,
            'max_age': self.max_age,
            'description': self.description,
            'expected_yield': self.expected_yield
        }

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user_message = db.Column(db.Text)
    ai_response = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)