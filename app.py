from flask import Flask, render_template,request, redirect
from helper import preprocessing, vectorizer, get_prediction


app = Flask(__name__)

data = dict()
reviews = []
positive = 0
negative = 0

@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html', data=data)

@app.route("/", methods = ['post'])
def my_post():
    text = request.form['text']
    preprocessed_txt = preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt)
    prediction = get_prediction(vectorized_txt)

    if prediction == 'negative':
        global negative
        negative += 1
    else:
        global positive
        positive += 1
    
    reviews.insert(0, text)
    return redirect(request.url)
    

if __name__ == "__main__":
    app.run()