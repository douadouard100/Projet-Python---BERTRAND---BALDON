import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    new = final_features[0][5]-final_features[0][6]
    final_features[0] = np.insert(final_features[0], 8, new)
    final_features[0] = np.insert(final_features, 12, 24)
    final = np.array([final_features[0], final_features[0]])
    prediction = model.predict(final)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Number of comments in the next 24 hours should be {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)