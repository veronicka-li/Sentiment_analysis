from flask import Flask, render_template, request

from model import classifier
from preprocess import get_clean_tokens, sent_vec

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
       text = request.form['input_text']
       pred = get_clean_tokens(text)
       predv = sent_vec(pred)
       result = classifier.predict(predv.reshape(1, -1))
       if result == [1] :
              out = 'Положительный отзыв'
       else:
              out = 'Отрицательный отзыв'
       return render_template('index.html',  final=out, text = text)
