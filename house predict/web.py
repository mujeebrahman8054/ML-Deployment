from flask import Flask,render_template,request
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('modelpkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
        areas=[int(x) for x in request.form.values()]
        area=np.array(areas).reshape(1,2)
        output=model.predict(areas)
        output=output.item()
        output=round(output,2)
        return render_template('result.html',prediction_text="price of the house is {}".format(output))
if __name__=='__main__':
     app.run(port=8000)    