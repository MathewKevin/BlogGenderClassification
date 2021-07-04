from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict_fun():
	
    NB_blog_model = open('NB_blog_model.pkl','rb')
    clf = joblib.load(NB_blog_model)
    
    tf_model = open('tfvec.pkl', 'rb')
    tf = joblib.load(tf_model)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tf.transform(data).toarray()
        my_prediction = clf.predict(vect)
	
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run()