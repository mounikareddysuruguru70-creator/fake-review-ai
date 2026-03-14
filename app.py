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



# Admin panel
@app.route("/admin")
def admin():

    conn=sqlite3.connect("reviews.db")

    reviews=conn.execute("SELECT * FROM history").fetchall()

    conn.close()

    return render_template("admin.html",reviews=reviews)



# Dashboard
@app.route("/dashboard")
def dashboard():

    import dashboard

    return render_template("dashboard.html")



if __name__=="__main__":
    app.run(debug=True)