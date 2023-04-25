from flask import Flask, render_template, request, app, jsonify, url_for
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
classifier=pickle.load(open('classifier.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data)
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = classifier.predict(new_data)
    output = int(output > 0.5)
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
    