from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

directory = "D:/VIEH Internship/Project/train/Dataset/updated-Employee-Attrition.csv"
dataset = pd.read_csv(directory)
model = pickle.load(open("D:/VIEH Internship/Project/saved_models/final_model.pkl", 'rb'))


def get_prediction(age, jobSatisfaction, monthlyIncome, totalWorkingYears, 
               yearsSinceLastPromotion, businessTravel, department, gender, 
               jobRole, maritalStatus):
    
    bt_i, d_i, g_i, jr_i, ms_i = -1,-1,-1,-1,-1
        
    if businessTravel != None:
        bt_i = np.where(dataset.columns == businessTravel)[0][0]
    
    if department != None:
        d_i = np.where(dataset.columns == department)[0][0]
        
    if gender != None:        
        g_i = np.where(dataset.columns == gender)[0][0]
        
    if jobRole != None:        
        jr_i = np.where(dataset.columns == jobRole)[0][0]

    if maritalStatus != None:        
        ms_i = np.where(dataset.columns == maritalStatus)[0][0]
            
    x = np.zeros(len(dataset.columns))
    x[0] = age
    x[1] = jobSatisfaction
    x[2] = monthlyIncome
    x[3] = totalWorkingYears
    x[4] = yearsSinceLastPromotion

    if bt_i >= 0:
        x[bt_i] = 1
    if d_i >= 0:
        x[d_i] = 1
    if g_i >= 0:
        x[g_i] = 1
    if jr_i >= 0:
        x[jr_i] = 1
    if ms_i >= 0:
        x[ms_i] = 1
        
    return model.predict([x])[0]


@app.route('/', methods=['GET'])
def hello():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    age = request.form['age']
    jobSatisfaction = request.form['jobSatisfaction']
    monthlyIncome = request.form['monthlyIncome']
    totalWorkingYears = request.form['totalWorkingYears']
    yearsSinceLastPromotion = request.form['yearsSinceLastPromotion']
    businessTravel = request.form['businessTravel']
    department = request.form['department']
    gender = request.form['gender']
    jobRole = request.form['jobRole']
    maritalStatus = request.form['maritalStatus']

    prediction = get_prediction(age, jobSatisfaction, monthlyIncome, totalWorkingYears, 
                                yearsSinceLastPromotion, businessTravel, department, 
                                gender, jobRole, maritalStatus)

    return render_template("pass.html", p=prediction)


if __name__ == "__main__":
    app.run(port=4000, debug=True)
