import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template,request

application = Flask(__name__)
app=application

##web application to the model

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])

def predict_datapoint():
    if request.method=="POST":
        #to get the datas input parameters
        Temperature=float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('WS'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))
        ##1st we do standard sacler
        new_data_scaled=standard_scaler.transform([[Temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge_model.predict(new_data_scaled)
        #//displaying the result which is trained in application.py
        return render_template('home.html',results=result[0])


        
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)

# or
# if __name__ == '__main__':
#     app.run(host="0.0.0.0")
