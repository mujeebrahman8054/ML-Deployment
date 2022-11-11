from flask import Flask,render_template,request
import numpy as np
import joblib
app=Flask(__name__)
model=joblib.load('model_joblib.h5')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    print(int_features)
    final_features=np.array(int_features).reshape(1,16)
    print(final_features)
    prediction=model.predict(final_features)
    L_collection={0:"Not fraudulent",1:"fraudulent"}
    result=L_collection[prediction[0]]
    print(result)
    return render_template('result.html',prediction_text=f"The job Application is {result}")
if __name__=="__main__":
    app.run(port=8000)      
