import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sqlite3
from wordcloud import WordCloud, STOPWORDS


def generate_charts():

    conn = sqlite3.connect("reviews.db")
    cursor = conn.cursor()

    # -------------------------
    # Fake vs Genuine Chart
    # -------------------------

    cursor.execute("SELECT prediction FROM history")
    data = cursor.fetchall()

    fake = 0
    genuine = 0

    for row in data:
        if row[0] == "Fake":
            fake += 1
        else:
            genuine += 1

    labels = ["Fake Reviews", "Genuine Reviews"]
    values = [fake, genuine]

    if fake + genuine > 0:
        plt.figure()
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title("Fake vs Genuine Reviews")
        plt.savefig("static/review_chart.png")
        plt.close()

    # -------------------------
    # Sentiment Chart
    # -------------------------

    cursor.execute("SELECT sentiment FROM history")
    sentiments = cursor.fetchall()

    pos = 0
    neg = 0
    neu = 0

    for row in sentiments:
        if row[0] == "Positive":
            pos += 1
        elif row[0] == "Negative":
            neg += 1
        else:
            neu += 1

    labels = ["Positive", "Negative", "Neutral"]
    values = [pos, neg, neu]

    if pos + neg + neu > 0:
        plt.figure()
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title("Sentiment Analysis")
        plt.savefig("static/sentiment_chart.png")
        plt.close()

    # -------------------------
    # WordCloud Generation
    # -------------------------

    cursor.execute("SELECT review FROM history ORDER BY id DESC LIMIT 50")
    reviews = cursor.fetchall()

    text = ""

    for row in reviews:
        text += row[0] + " "

    # Remove common meaningless words
    stop_words = set(STOPWORDS)
    extra_words = {
        "phone","product","money","buy","use","amazon",
        "flipkart","item","good","bad","waste","amazing",
        "very","really","best","worst","this","that","is",
        "was","are","the","a","an","it"
    }

    stop_words.update(extra_words)

    if text.strip() != "":
        wordcloud = WordCloud(
            width=900,
            height=450,
            background_color="white",
            stopwords=stop_words,
            max_words=40
        ).generate(text)

        wordcloud.to_file("static/wordcloud.png")

    conn.close()

    print("Charts created successfully")