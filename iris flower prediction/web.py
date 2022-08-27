from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('modelpk','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    iris_features=[float(x) for x in request.form.values()]
    entered_features=np.array(iris_features).reshape(1,4)
    prediction=model.predict(entered_features)
    prediction_values={0:"setosa",1:"versicolor",2:"virginica"}
    result=prediction_values[prediction[0]]
    return render_template('results.html',prediction_text="it belongs to {}".format(result))
if __name__=='__main__':
    app.run(port=8000)    
