import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("data/Titanic-Dataset.csv")

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features & target
X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df['Survived']

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save train/test split for later testing
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/titanic_model.pkl")
joblib.dump((X_val, y_val), "artifacts/test_data.pkl")

print("✅ Model trained and saved in artifacts/")
