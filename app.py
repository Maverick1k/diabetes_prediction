from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

smoking_history_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = smoking_history_mapping[request.form['smoking_history']]
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = int(request.form['blood_glucose_level'])
        
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [hba1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

        print("Input data received:")
        print(input_data)
        
        input_data_scaled = scaler.transform(input_data)
        
        prob = model.predict_proba(input_data_scaled)[0]
        
        threshold = 0.6
        prediction = 1 if prob[1] > threshold else 0
        
        if prediction == 1:
            result = "The model predicts that you may have diabetes."
        else:
            result = "The model predicts that you are not likely to have diabetes."
    
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)