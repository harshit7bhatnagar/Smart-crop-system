import os
from dotenv import load_dotenv

load_dotenv()

# Default to SQLite for immediate local runs.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///farm.db")

# Model path
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "crop_model.pkl"))
