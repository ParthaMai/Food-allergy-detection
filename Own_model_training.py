import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("modified_final_food_allergen.csv")

# Fill missing values in label columns
df = df.fillna(0)

# Input
X = df["Ingredients"].astype(str)

# Output labels
y = df[[
    "peanut", "almond", "pistachio", "cashew",
    "milk", "butter","cheese", "paneer", "ghee",
    "egg",
    "fish", "prawn",
    "moong dal", "chana dal",
    "tomato", "banana",
    "capsicum", "mushroom", "bitter gourd",
    "mustard",
    "chocolate",
    "chicken", "mutton"
]]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "allergy_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Training Complete")