from flask import Flask,render_template,request
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('modelbc.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    print(int_features)
    final_features=np.array(int_features).reshape(1,9)
    print(final_features)
    prediction=model.predict(final_features)
    L_collection={0:"No Reccurence Event", 1:"Reccurence Event"}
    result=L_collection[prediction[0]]
    print(result)
    return render_template('result.html',prediction_text=f"The Breast Cancer is :{result}")
if __name__=='__main__':
    app.run(port=8000)      