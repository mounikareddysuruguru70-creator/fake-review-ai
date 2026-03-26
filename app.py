from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Model accuracy for display (hardcoded or read from file)
# Replace 85 with your model's actual test accuracy if available
model_accuracy = 85  

# Store review history in memory
review_history = []

# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form.get('review')
        if review:
            return redirect(url_for('predict', review_text=review))
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        review = request.form.get('review')
    else:
        review = request.args.get('review_text')

    if not review:
        return redirect(url_for('home'))

    # Prediction
    prediction = model.predict([review])[0]
    confidence = max(model.predict_proba([review])[0]) * 100

    # Sentiment analysis
    polarity = TextBlob(review).sentiment.polarity
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Add to history
    review_history.append({
        'review': review,
        'prediction': prediction,
        'confidence': round(confidence,2),
        'sentiment': sentiment,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    # Generate dynamic word cloud
    text = " ".join([r['review'] for r in review_history])
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='plasma', max_words=200).generate(text)
    wc.to_file('static/wordcloud.png')

    # Generate colorful bar graph
    df = pd.DataFrame(review_history)
    counts = df['prediction'].value_counts()
    plt.figure(figsize=(8,5))
    colors = ['#FF5733', '#33FF57', '#3357FF']
    counts.plot(kind='bar', color=colors[:len(counts)])
    plt.title('Fake vs Genuine Reviews', fontsize=16)
    plt.xlabel('Review Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    for index, value in enumerate(counts):
        plt.text(index, value, str(value), ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/graph.png')
    plt.close()

    return render_template('result.html', review=review, prediction=prediction,
                           confidence=round(confidence,2), sentiment=sentiment,
                           model_accuracy=model_accuracy)

# Review history
@app.route('/history', methods=['GET'])
def history():
    return render_template('history.html', reviews=review_history)

# CSV upload
@app.route('/upload', methods=['GET','POST'])
def upload_csv():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            results = []
            for review in df['review']:
                pred = model.predict([review])[0]
                conf = max(model.predict_proba([review])[0]) * 100
                results.append({'review': review, 'prediction': pred, 'confidence': round(conf,2)})
            return render_template('history.html', reviews=results)
    return render_template('upload.html')

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)