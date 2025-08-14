import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Load model and test data
model = joblib.load("artifacts/titanic_model.pkl")
X_val, y_val = joblib.load("artifacts/test_data.pkl")

# Predictions
y_pred = model.predict(X_val)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Save metrics
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

print("✅ Metrics calculated and saved in artifacts/metrics.txt")
