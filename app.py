from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, 'D:\projectsCertificates\stockSentiment\stocks\Lib\site-packages')

file1 = 'stockSentiment.pkl'
rfClassifier = pickle.load(open(file1, 'rb'))

file2 = 'countVec.pkl'
countVectorizer = pickle.load(open(file2, 'rb'))

app = Flask(__name__)

@app.route('/', methods = ["GET"])
def home():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == 'POST':
        headlineFile = request.form['headline']
        data = pd.read_csv(headlineFile)
        mod_data = []
        mod_data.append(' '.join(str(x) for x in data.iloc[0:25]))
        vector = countVectorizer.transform(mod_data).toarray()
        prediction = rfClassifier.predict(vector)
        if prediction == 0:
            return render_template('index.html', prediction_text = 'The stock price will not change')
        elif prediction == 1:
            return render_template('index.html', prediction_text = 'The stock price will change')
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
