# Crop Recommendation Backend ("Spine")

This repo contains:
- **Step 1 (Brain)**: Training script to build `crop_model.pkl` from a Kaggle-like dataset.
- **Step 2 (Spine)**: A Flask API that loads the trained model and exposes prediction & basic CRUD endpoints.
- A **demo model** is included (synthetic) so you can run `/predict` immediately. For real use, retrain with the Kaggle dataset.

## ✅ Quick Start (Local, no Postgres needed)

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Initialize the DB (SQLite by default)
```bash
python backend/migrate.py
```

### 4) Run the API
```bash
python backend/app.py
```
Server starts at `http://127.0.0.1:5000`.

### 5) Test prediction
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90, "P": 42, "K": 43,
    "temperature": 20, "humidity": 80,
    "ph": 6.5, "rainfall": 200
  }'
```

## 🔁 Train Your Own Model (Step 1)

1) Download the Kaggle **Crop Recommendation** dataset (CSV with columns: `N,P,K,temperature,humidity,ph,rainfall,label`).  
2) Place it anywhere (e.g., `data/crop_recommendation.csv`).  
3) Train:
```bash
python crop_brain/train_model.py --csv data/crop_recommendation.csv --out backend/crop_model.pkl
```
4) Restart the API.

## 🗄️ Switch to PostgreSQL (Optional)

1) Create a DB and set env var `DATABASE_URL` before running:
```
DATABASE_URL=postgresql://<user>:<pass>@localhost:5432/farm_db
```
2) Recreate tables:
```bash
python backend/migrate.py
```

## 📚 Endpoints

- `GET /health` → health check
- `POST /predict` → predict crop
- `POST /register_farmer` → add farmer
- `POST /log_fertilizer` → add fertilizer record
- `GET /farmers/<id>` → fetch a farmer + crops + logs

See `backend/app.py` for request/response schemas.

## 🔐 Notes
- CORS is enabled for ease of frontend integration.
- Replace the demo model with your own for production.
