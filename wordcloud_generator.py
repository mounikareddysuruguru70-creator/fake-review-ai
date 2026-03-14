import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt

conn = sqlite3.connect("reviews.db")

data = conn.execute("SELECT review FROM history").fetchall()

conn.close()

text = " ".join([i[0] for i in data])

if text == "":
    text = "No Reviews Yet"

wordcloud = WordCloud(width=800,height=400,background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")

plt.savefig("static/wordcloud.png")

print("WordCloud created")