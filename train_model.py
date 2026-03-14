import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("data/reviews.csv")

# Text cleaning
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    return text

df["review"] = df["review"].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["review"])
y = df["label"]

# Train model
model = SVC()

model.fit(X, y)

# Save model
pickle.dump((model,vectorizer),open("model/fake_review_model.pkl","wb"))

print("Model trained successfully")