from flask import Flask
from flask import redirect, url_for
from flask import request

import pandas as pd
import numpy as np

from joblib import dump, load

# Loading model 
reg = load('regression_model.joblib')

def reg_model(v1, v2, v3):
    test = np.array([v1,v2,v3]).reshape(1,-1)
    predicted_value = reg.predict(test)
    return predicted_value*100


# flask object creation
app = Flask(__name__)

# basic use of route
@app.route('/')
def hello_world():
   return 'Hello World'
 
# Passing data to the model for prediction
@app.route('/regression/predict/<value1>/<value2>/<value3>')
def prediction(value1, value2, value3):
    chance = reg_model(value1, value2,value3)
    return "Chances of getting into the university is %2f" %chance
        
# getting the data from HTML page
@app.route('/regression', methods = ['POST','GET'])
def get_data():
    #request.method == 'POST'
    gre = request.form['gre']
    cgpa = request.form['cgpa']
    toefl = request.form['toefl']
    return redirect(url_for('prediction',value1 = gre, value2 = cgpa, value3 = toefl))

if __name__ == '__main__':
   app.run()