import pickle
import re
from textblob import TextBlob

model = pickle.load(open("model/model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    return text


def get_sentiment(review):

    analysis = TextBlob(review)

    if analysis.sentiment.polarity > 0:
        return "Positive"

    elif analysis.sentiment.polarity < 0:
        return "Negative"

    else:
        return "Neutral"


def predict_review(review):

    cleaned = clean_text(review)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    sentiment = get_sentiment(review)

    return prediction, sentiment