from flask import Flask,render_template,request
import sqlite3
import pickle
from textblob import TextBlob

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load ML model
model,vectorizer = pickle.load(open("model/fake_review_model.pkl","rb"))


# Create database
conn = sqlite3.connect("reviews.db")
conn.execute("""
CREATE TABLE IF NOT EXISTS history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
review TEXT,
prediction TEXT,
sentiment TEXT,
rating INTEGER
)
""")
conn.close()


# Spam keywords
spam_words = [
"buy now",
"limited offer",
"best deal",
"click here",
"100% guarantee"
]


# Duplicate review detection
def check_similarity(new_review):

    conn = sqlite3.connect("reviews.db")

    old_reviews = conn.execute("SELECT review FROM history").fetchall()

    conn.close()

    old_reviews = [r[0] for r in old_reviews]

    if len(old_reviews) == 0:
        return False

    reviews = [new_review] + old_reviews

    cv = CountVectorizer()

    matrix = cv.fit_transform(reviews)

    similarity = cosine_similarity(matrix)

    if similarity[0][1:].max() > 0.8:
        return True

    return False



# Home page
@app.route("/",methods=["GET","POST"])
def home():

    prediction=None
    sentiment=None

    if request.method=="POST":

        review=request.form["review"]
        rating=request.form["rating"]

        # Vectorize review
        review_vector=vectorizer.transform([review])

        pred=model.predict(review_vector)[0]

        if pred==1:
            prediction="Fake"
        else:
            prediction="Genuine"

        # Spam keyword detection
        for word in spam_words:
            if word in review.lower():
                prediction="Fake"

        # Duplicate detection
        if check_similarity(review):
            prediction="Fake"

        # Sentiment analysis
        polarity=TextBlob(review).sentiment.polarity

        if polarity>0:
            sentiment="Positive"
        elif polarity<0:
            sentiment="Negative"
        else:
            sentiment="Neutral"

        # Save to database
        conn=sqlite3.connect("reviews.db")

        conn.execute(
        "INSERT INTO history(review,prediction,sentiment,rating) VALUES (?,?,?,?)",
        (review,prediction,sentiment,rating)
        )

        conn.commit()
        conn.close()

    return render_template("index.html",
                           prediction=prediction,
                           sentiment=sentiment)



```python
from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data (only first time)
nltk.download('punkt')

app = Flask(__name__)

# ---------------------------
# Sample Dataset (you can replace with your CSV later)
# ---------------------------
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
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Genuine, 0 = Fake
}

df = pd.DataFrame(data)

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned'] = df['review'].apply(clean_text)

# ---------------------------
# Feature Extraction
# ---------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# ---------------------------
# Model Training
# ---------------------------
model = LogisticRegression()
model.fit(X, y)

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "Genuine Review ✅"
    else:
        result = "Fake Review ❌"

    return render_template('index.html', prediction_text=result)

# ---------------------------
# Run App (IMPORTANT for deployment)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
```
