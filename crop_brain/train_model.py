import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import lightgbm as lgb

# ---------------------------
# 1. Load Dataset
# ---------------------------
data = pd.read_csv("data/crop_recommendation.csv")

# Features & Labels
X = data.drop("Crop", axis=1)
y = data["Crop"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 2. Define Models
# ---------------------------
models = {
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(criterion="entropy", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

results = {}
trained_models = {}

# ---------------------------
# 3. Train & Evaluate Models
# ---------------------------
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model

    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, fmt="g", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix (Accuracy: {acc:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ---------------------------
# 4. Compare Models
# ---------------------------
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color="skyblue")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ---------------------------
# 5. Save Best Model
# ---------------------------
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

joblib.dump({"model": best_model, "feature_order": list(X.columns)}, "best_model.pkl")

print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
print("✅ Saved best model as best_model.pkl")