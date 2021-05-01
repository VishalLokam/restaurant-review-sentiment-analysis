from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import joblib
import pickle
import json


filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home_page():
    return 'Home page'


@app.route('/predict', methods=['POST'])
def predict():
    positiveReview = 0
    negativeReview = 0
    count = 0
    if request.method == 'POST':
        content = request.get_json(silent=False)
        for i in content['restaurant-reviews']:
            count = count+1
            print(i['review'])
            data = [i['review']]
            vect = cv.transform(data).toarray()
            my_prediction = clf.predict(vect)
            if my_prediction == 1:
                positiveReview = positiveReview+1
            elif my_prediction == 0:
                negativeReview = negativeReview+1

        positivePercentage = (positiveReview/count)*100
        negativePercentage = (negativeReview/count)*100
        data_summary = {"positive": positiveReview, "negative": negativeReview,
                        "positive-percentage": positivePercentage, "negative-percentage": negativePercentage}
        json_dump = json.dumps(data_summary)

        # print(type(data_summary))
    # return "Positive:"+str(positiveReview)+" Negative:"+str(negativeReview)
    return json_dump


if __name__ == '__main__':
    app.run(debug=True)
