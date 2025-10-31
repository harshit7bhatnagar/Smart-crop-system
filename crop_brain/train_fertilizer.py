import pandas as pd
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
data = pd.read_csv("data/fertilizer_training_data_updated.csv")  # use your uploaded dataset

# Features & Labels
X = data.drop("Fertilizer", axis=1)
y = data["Fertilizer"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 2. Define Models
# ---------------------------
models = {
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(criterion="entropy", random_state=42, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
}

results = {}
trained_models = {}

# ---------------------------
# 3. Train & Evaluate Models
# ---------------------------
for name, model in models.items():
    print(f"\n🔹 Training {name} for Fertilizer...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model

    print(f"{name} Accuracy (Fertilizer): {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=False, fmt="g", cmap="Blues")
    plt.title(f"{name} - Fertilizer Confusion Matrix (Accuracy: {acc:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ---------------------------
# 4. Compare Models
# ---------------------------
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color="skyblue")
plt.ylabel("Accuracy")
plt.title("Fertilizer Model Accuracy Comparison")
plt.xticks(rotation=30)
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison_fertilizer.png")
plt.show()

# ---------------------------
# 5. Save Best Model
# ---------------------------
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

joblib.dump({
    "model": best_model,
    "feature_order": list(X.columns),
    "classes": best_model.classes_
}, "best_model_fertilizer.pkl")

print(f"\n✅ Best model for Fertilizer: {best_model_name} with accuracy {results[best_model_name]:.4f}")
print("✅ Saved as best_model_fertilizer.pkl")
