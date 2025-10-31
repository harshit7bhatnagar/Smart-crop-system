from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import sqlite3
import os
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# ---------------- Load Models ----------------
# Crop model comes from train_model.py → best_model.pkl
crop_model_data = joblib.load("best_model.pkl")
crop_model = crop_model_data["model"]
crop_features = crop_model_data["feature_order"]

# If "classes" is missing, derive from model directly
if hasattr(crop_model, "classes_"):
    crop_classes = crop_model.classes_
else:
    crop_classes = None

# Fertilizer model comes from train_fertilizer.py → best_model_fertilizer.pkl
fert_model_data = joblib.load("best_model_fertilizer.pkl")
fert_model = fert_model_data["model"]
fert_features = fert_model_data["feature_order"]
fert_classes = fert_model_data["classes"]

app = Flask(__name__)
CORS(app, supports_credentials=True)

# ----------------- Simple SQLite auth ------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- Auth endpoints ----------------
@app.route("/api/register", methods=["POST"])
def api_register():
    try:
        body = request.get_json() or {}
        name = (body.get("name") or "").strip()
        email = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""

        if not name or not email or not password:
            return jsonify({"success": False, "message": "name, email and password are required"}), 400

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            conn.close()
            return jsonify({"success": False, "message": "Email already registered"}), 409

        pw_hash = generate_password_hash(password)
        now = datetime.utcnow().isoformat()
        cur.execute("INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (name, email, pw_hash, now))
        conn.commit()
        user_id = cur.lastrowid
        conn.close()

        return jsonify({"success": True, "user": {"id": user_id, "name": name, "email": email}})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/login", methods=["POST"])
def api_login():
    try:
        body = request.get_json() or {}
        email = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""
        if not email or not password:
            return jsonify({"success": False, "message": "email and password required"}), 400

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, name, email, password_hash, created_at FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

        if not check_password_hash(row["password_hash"], password):
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

        user = {"id": row["id"], "name": row["name"], "email": row["email"]}
        return jsonify({"success": True, "user": user})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/user", methods=["GET"])
def api_user():
    return jsonify({"success": True, "message": "Use /api/login to authenticate"})

@app.route("/api/logout", methods=["GET"])
def api_logout():
    return jsonify({"success": True})

# ---------------- Prediction endpoints ----------------
@app.route("/")
def home():
    return {"message": "Crop Recommendation API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        mapping = {
            "temperature": "Temperature",
            "moisture": "Moisture",
            "rainfall": "Rainfall",
            "ph": "PH",
            "nitrogen": "Nitrogen",
            "phosphorous": "Phosphorous",
            "potassium": "Potassium",
            "carbon": "Carbon",
            "soil": "Soil"
        }

        # -------- Crop Prediction --------
        crop_input = {mapping[k]: v for k, v in data.items() if k in mapping}
        missing = [f for f in crop_features if f not in crop_input]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        crop_df = pd.DataFrame([crop_input], columns=crop_features)

        if hasattr(crop_model, "predict_proba"):
            crop_probs = crop_model.predict_proba(crop_df)[0]
            crop_top_idx = np.argsort(crop_probs)[::-1][:3]
            if crop_classes is not None:
                crop_top3 = [
                    {"label": str(crop_classes[i]), "probability": float(crop_probs[i])}
                    for i in crop_top_idx
                ]
            else:
                # fallback: just indices if no classes
                crop_top3 = [
                    {"label": f"Class_{i}", "probability": float(crop_probs[i])}
                    for i in crop_top_idx
                ]
        else:
            pred_crop = crop_model.predict(crop_df)[0]
            crop_top3 = [{"label": str(pred_crop), "probability": 1.0}]

        predicted_crop = crop_top3[0]["label"]

        # -------- Fertilizer Prediction --------
        fert_input = crop_input.copy()
        fert_df = pd.DataFrame([fert_input], columns=[c for c in fert_features if not c.startswith("Crop_")])

        # Add crop one-hot features
        for col in fert_features:
            if col.startswith("Crop_"):
                fert_df[col] = 1 if col == f"Crop_{predicted_crop}" else 0

        fert_df = fert_df[fert_features]  # ensure correct column order

        if hasattr(fert_model, "predict_proba"):
            fert_probs = fert_model.predict_proba(fert_df)[0]
            fert_top_idx = np.argsort(fert_probs)[::-1][:3]
            fert_top3 = [
                {"label": str(fert_classes[i]), "probability": float(fert_probs[i])}
                for i in fert_top_idx
            ]
        else:
            pred_fert = fert_model.predict(fert_df)[0]
            fert_top3 = [{"label": str(pred_fert), "probability": 1.0}]

        return jsonify({
            "crop": {"top1": crop_top3[0], "top3": crop_top3},
            "fertilizer": {"top1": fert_top3[0], "top3": fert_top3},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
