import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

df = pd.read_csv("modified_final_food_allergen.csv")

X = df["Ingredients"].astype(str)

y = df[[
    "peanut","almond","pistachio","cashew",
    "milk","butter","cheese","paneer","ghee",
    "egg","fish","prawn",
    "moong dal","chana dal",
    "tomato","banana",
    "capsicum","mushroom","bitter gourd",
    "mustard","chocolate","chicken","mutton"
]]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42
)
prediction_times = []

models = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(max_iter=1000)
    ),

    "Random Forest": OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    ),

    "Linear SVM": OneVsRestClassifier(
        LinearSVC()
    )
}

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Measure average prediction time
    start = time.time()

    for _ in range(100):
        model.predict(X_test)

    end = time.time()

    avg_time = (end - start) / 100
    prediction_times.append(avg_time)

# Plot
plt.figure(figsize=(8,5))
plt.bar(models.keys(), prediction_times)

plt.ylabel("Average Prediction Time (seconds)")
plt.title("Prediction Time Comparison")

for i, v in enumerate(prediction_times):
    plt.text(i, v, f"{v:.5f}", ha="center")

plt.show()

import time

training_times = []

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    training_times.append(end-start)

plt.figure(figsize=(8,5))
plt.bar(models.keys(), training_times)
plt.ylabel("Seconds")
plt.title("Training Time Comparison")
plt.show()

import os
import joblib

sizes=[]

for name,model in models.items():

    model.fit(X_train,y_train)

    filename=name.replace(" ","_")+".pkl"

    joblib.dump(model,filename)

    sizes.append(os.path.getsize(filename)/1024)

plt.figure(figsize=(8,5))
plt.bar(models.keys(),sizes)
plt.ylabel("KB")
plt.title("Model Size Comparison")
plt.show()