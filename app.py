
from flask import Flask, render_template, request, send_file
import pandas as pd
import re
import nltk
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('punkt')

app = Flask(__name__)

# ---------------- Dataset ----------------
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

# ---------------- Cleaning ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned'] = df['review'].apply(clean_text)

# ---------------- Model ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

accuracy = accuracy_score(y, model.predict(X))

# ---------------- Storage ----------------
prediction_history = []
review_history = []

# ---------------- Graph ----------------
def generate_graph():
    fake = prediction_history.count(0)
    genuine = prediction_history.count(1)

    plt.figure()
    plt.bar(['Fake', 'Genuine'], [fake, genuine])
    plt.title("Prediction Graph")

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/graph.png")
    plt.close()

# ---------------- Word Cloud ----------------
def generate_wordcloud():
    text = " ".join(review_history) if review_history else "No Data"
    wc = WordCloud(width=400, height=200).generate(text)
    wc.to_file("static/wordcloud.png")

# ---------------- Home ----------------
@app.route('/')
def home():
    generate_graph()
    generate_wordcloud()
    return render_template('index.html',
                           accuracy=round(accuracy*100,2),
                           history=review_history[-5:])

# ---------------- Predict ----------------
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')

    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    prediction_history.append(prediction)
    review_history.append(review)

    prob = model.predict_proba(vector)[0]
    confidence = max(prob) * 100

    result = "Genuine Review ✅" if prediction==1 else "Fake Review ❌"
    color = "green" if prediction==1 else "red"

    generate_graph()
    generate_wordcloud()

    return render_template('index.html',
                           prediction_text=result,
                           confidence=f"{confidence:.2f}%",
                           color=color,
                           accuracy=round(accuracy*100,2),
                           history=review_history[-5:])

# ---------------- CSV Upload ----------------
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    data = pd.read_csv(file)

    results = []
    for review in data['review']:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]
        results.append("Genuine" if pred==1 else "Fake")

    data['Prediction'] = results
    data.to_csv("static/output.csv", index=False)

    return render_template('index.html',
                           prediction_text="CSV Processed ✅",
                           download=True,
                           accuracy=round(accuracy*100,2),
                           history=review_history[-5:])

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

