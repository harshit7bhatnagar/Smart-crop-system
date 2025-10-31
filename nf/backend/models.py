from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Farmer(db.Model):
    __tablename__ = "farmers"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    location = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    crops = db.relationship("Crop", backref="farmer", lazy=True, cascade="all, delete-orphan")
    logs = db.relationship("FertilizerLog", backref="farmer", lazy=True, cascade="all, delete-orphan")

class Crop(db.Model):
    __tablename__ = "crops"
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmers.id"), nullable=False)
    crop_name = db.Column(db.String(80), nullable=False)
    planted_date = db.Column(db.Date, nullable=True)

    logs = db.relationship("FertilizerLog", backref="crop", lazy=True, cascade="all, delete-orphan")

class FertilizerLog(db.Model):
    __tablename__ = "fertilizer_logs"
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmers.id"), nullable=False)
    crop_id = db.Column(db.Integer, db.ForeignKey("crops.id"), nullable=True)
    date = db.Column(db.Date, nullable=False)
    fertilizer_type = db.Column(db.String(80), nullable=False)
    quantity = db.Column(db.Float, nullable=True)
    notes = db.Column(db.String(255), nullable=True)
