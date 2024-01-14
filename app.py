from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

fertility_model = pickle.load(open('fertility_model.pkl', 'rb'))
crop_model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
fertilizer_model = pickle.load(open('fertilizer_recommendation_model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/fertility',methods = ['POST', 'GET'])
def fertility():
    
    if request.method == 'POST':
        form = request.form.to_dict()

        X = np.array([[
            float(form["clay"]),
            float(form["sand"]),
            float(form["cec"]),
            float(form["caco3"]),
            float(form["fe"])
        ]])
        X = pd.DataFrame(X, columns=['Clay', 'Sand', 'CEC', 'CaCO3', 'Fe'])
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=['Clay', 'Sand', 'CEC', 'CaCO3', 'Fe'])

        result = fertility_model.predict(X)[0]
        response = "Fertile" if result == 1 else "Non Fertile"
        data = {'form': form, 'response': response}
        print(response)
        return render_template("fertility.html", data=data)

    return render_template("fertility.html")

@app.route('/crop',methods = ['POST', 'GET'])
def crop():
    if request.method == 'POST':
        form = request.form.to_dict()

        X = np.array([[
            float(form["humidity"]),
            float(form["potassium"]),
            float(form["rainfall"]),
            float(form["phosphorous"]),
            float(form["temperature"])
        ]])
        X = pd.DataFrame(X, columns=['humidity', 'K', 'rainfall', 'P', 'temperature'])

        response = crop_model.predict(X)[0]
        data = {'form': form, 'response': response}
        print(response)
        return render_template("crop.html", data=data)

    return render_template("crop.html")

@app.route('/fertilizer',methods = ['POST', 'GET'])
def fertilizer():
    
    if request.method == 'POST':
        form = request.form.to_dict()

        X = np.array([[
            float(form["phosphorous"]),
            float(form["nitrogen"]),
            float(form["potassium"])
        ]])

        X = pd.DataFrame(X, columns=['Phosphorous', 'Nitrogen', 'Potassium'])
        
        response = fertilizer_model.predict(X)[0]
        data = {'form': form, 'response': response}
        print(response)
        return render_template("fertilizer.html", data=data)

    return render_template("fertilizer.html")


if __name__ == '__main__':
    app.run(debug = True)