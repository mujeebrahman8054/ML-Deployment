from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model (1).pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    yearexp=float(request.values['yearexp'])
    yearexp=np.reshape(yearexp,(-1,1))
    output=model.predict(yearexp)
    output=output.item()
    output=round(output,2)
    return render_template('results.html',prediction_text="salary for this years of experience is {}".format(output))
if __name__=='__main__':
    app.run(port=8000)      