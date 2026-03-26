from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

app = Flask(__name__)

# Dataset
data = {
    "review": [
        "This product is amazing and works perfectly",
        "Worst product ever waste of money",
        "Absolutely fantastic quality highly recommend",
        "Fake product not as described",
        "Very good and useful product",
        "Terrible experience will not buy again",
        "Excellent item loved it",
        "Do not buy this it is fake"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned'] = df['review'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Model
model = LogisticRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review = request.form.get('review')

        if not review:
            return render_template('index.html', prediction_text="Enter a review")

        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        confidence = max(prob) * 100

        if prediction == 1:
            result = "Genuine Review ✅"
            color = "green"
        else:
            result = "Fake Review ❌"
            color = "red"

        return render_template(
            'index.html',
            prediction_text=result,
            confidence=f"{confidence:.2f}%",
            color=color
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)