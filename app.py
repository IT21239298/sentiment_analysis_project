from flask import Flask, render_template, request, redirect, jsonify
from flask_cors import CORS
from helper import preprocessing, vectorizer, get_prediction

app = Flask(__name__)
CORS(app)

data = {'review': None, 'decision': None}

@app.route("/")
def index():
    return redirect("http://localhost:3000/addreviews/:id")

@app.route("/analysis")
def analysis():
    return jsonify(data)

@app.route("/", methods=['POST'])
def my_post():
    text = request.form['text']
    preprocessed_txt = preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt)
    prediction = get_prediction(vectorized_txt)

    global data
    data['review'] = {'text': text}  # Only store the latest review
    data['decision'] = 'positive' if prediction != 'negative' else 'negative'  # Update the decision
    
    return jsonify(data)

if __name__ == "__main__":
    app.run()
