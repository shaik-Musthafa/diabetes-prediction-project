import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
        suggestions = get_food_suggestions(1)
    else:
        pred = "You don't have Diabetes."
        suggestions = get_food_suggestions(0)

    return render_template('index.html', prediction_text=pred, suggestions_text=suggestions)


def get_food_suggestions(prediction):
    if prediction == 1:
        suggestions = "'Fruits', 'Vegetables', 'Brown Rice'"
    else:
        suggestions = "'sugar candies', 'sugar items', 'sugar cane'"

    return suggestions


if __name__ == "__main__":
    app.run(debug=True)
