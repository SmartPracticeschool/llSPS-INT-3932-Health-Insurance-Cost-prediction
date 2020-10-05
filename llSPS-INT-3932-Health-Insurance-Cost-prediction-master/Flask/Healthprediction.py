import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model= load('model.save')
trans=load('transf')

@app.route('/')
def home():
    return render_template('Frontend1.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    test=trans.transform(x_test)
    test=test[:,1:]
    print(test)
    prediction = model.predict(test)
    print(prediction)
    output=prediction[0]
    
    return render_template('Frontend1.html', prediction_text='Charges {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
