from flask import *
import joblib
from flask import request
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome To Diabetes APP"
@app.route("/Diabetes")
def diabetes():
    return render_template("diabetes.html")

def valudepredictior(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if (size == 8):
        loaded_model  = joblib.load('diabetes_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        if (len(to_predict_list)== 8):
            result = valudepredictior(to_predict_list,8)
    if (int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return (render_template('result.html', prediction_text = prediction))

if __name__ == '__main__':
    app.run(debug = True)

