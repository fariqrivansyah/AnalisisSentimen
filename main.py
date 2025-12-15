from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

# Load ANN model
model = load_model("sentiment_ann.h5")

# Load TF-IDF
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ------------------------
# Fungsi Preprocessing
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review_text = ""
    if request.method == "POST":
        review_text = request.form["review"]
        clean_input = clean_text(review_text)
        X_input = tfidf.transform([clean_input]).toarray()
        
        # ANN prediction
        pred_prob = model.predict(X_input)
        prediction = "Positif" if pred_prob > 0.5 else "Negatif"

    return render_template("index.html", prediction=prediction, review_text=review_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

