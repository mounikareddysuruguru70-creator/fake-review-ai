import os
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Ensure folders exist
# ---------------------------
os.makedirs("static", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)
os.makedirs("static/wordclouds", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ---------------------------
# Load model and vectorizer
# ---------------------------
model = joblib.load("model.pkl")  # Your trained ML model
vectorizer = joblib.load("vectorizer.pkl")  # CountVectorizer used in training

# ---------------------------
# In-memory review history
# ---------------------------
review_history = []

# ---------------------------
# Home Page
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------
# Single Review Prediction
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    # Vectorize review
    review_vect = vectorizer.transform([review])

    # Predict Fake/Genuine
    pred = model.predict(review_vect)[0]
    confidence = model.predict_proba(review_vect).max() * 100

    # Sentiment
    sentiment_score = TextBlob(review).sentiment.polarity
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Save review in history
    review_history.append({
        "review": review,
        "prediction": pred,
        "confidence": f"{confidence:.2f}%",
        "sentiment": sentiment
    })

    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color="white").generate(review)
    wc_file = "static/wordclouds/wordcloud.png"
    wc.to_file(wc_file)

    # Generate simple bar graph
    labels = ["Fake", "Genuine"]
    values = [
        sum(1 for r in review_history if r["prediction"] == "Fake"),
        sum(1 for r in review_history if r["prediction"] == "Genuine")
    ]
    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["red", "green"])
    plt.title("Review Predictions Count")
    graph_file = "static/graphs/graph.png"
    plt.savefig(graph_file)
    plt.close()

    return render_template("result.html",
                           review=review,
                           prediction=pred,
                           confidence=f"{confidence:.2f}%",
                           sentiment=sentiment,
                           wordcloud_path=wc_file,
                           graph_path=graph_file)

# ---------------------------
# CSV Upload & Batch Prediction
# ---------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        df = pd.read_csv(file_path)
        if "review" not in df.columns:
            return "CSV must have a 'review' column"

        reviews = df["review"].tolist()
        review_vectors = vectorizer.transform(reviews)
        predictions = model.predict(review_vectors)
        confidences = model.predict_proba(review_vectors).max(axis=1) * 100

        # Sentiment
        sentiments = []
        for review in reviews:
            score = TextBlob(review).sentiment.polarity
            if score > 0:
                sentiments.append("Positive")
            elif score < 0:
                sentiments.append("Negative")
            else:
                sentiments.append("Neutral")

        df["prediction"] = predictions
        df["confidence"] = [f"{c:.2f}%" for c in confidences]
        df["sentiment"] = sentiments

        # Append to history
        for i in range(len(df)):
            review_history.append({
                "review": df.loc[i, "review"],
                "prediction": df.loc[i, "prediction"],
                "confidence": df.loc[i, "confidence"],
                "sentiment": df.loc[i, "sentiment"]
            })

        return render_template("results.html", tables=[df.to_html(classes='data')], titles=df.columns.values)

    return render_template("upload.html")

# ---------------------------
# Review History Page
# ---------------------------
@app.route("/history")
def history():
    return render_template("history.html", reviews=review_history)

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)