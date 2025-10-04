import pickle
import os
import sys
import flask
from flask import Flask,render_template,request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
le = WordNetLemmatizer()
import string
import pickle
import os
import sys
app = Flask(__name__)
co = pickle.load(open('BOW_CV.pkl','rb'))
sol = pickle.load(open('BOW_ML.pkl','rb'))


@app.route('/')
def fun():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def fun2():
    if request.method == 'POST':
        text = request.form["message"]
        # clean the text
        text = text.lower()
        text = ''.join([j for j in text if j not in string.punctuation])  # remove punctuations
        text = ' '.join([k for k in text.split() if k not in stopwords.words('english')])  # remove stopwords
        text = ' '.join([le.lemmatize(p) for p in text.split()])
        d = [text]
        vectors = co.transform(d)
        vectors = vectors.toarray()
        final = sol.predict(vectors)
        final = final[0]
        if final == 0:
            return render_template('index.html',prediction_text = "Spam Mail")
        else:
            return render_template('index.html',prediction_text = "Inbox Mail")



if __name__ == '__main__':
    app.run(debug=True)