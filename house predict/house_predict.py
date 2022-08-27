import pandas as pd
import numpy as np
import pickle
data=pd.read_csv('homeprices (2).csv')
print(data)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ledata=data
ledata.town=le.fit_transform(ledata.town)
#data=data.drop('town',axis=1,inplace=True)
print(ledata)
x=ledata[['area','town']].values
y=ledata.price
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn import linear_model
lr=linear_model.LinearRegression()
model=lr.fit(x_train,y_train)
with open('modelpkl','wb') as file:
  pickle.dump(model,file)