import pandas as pd
import numpy as np
import pickle
data=pd.read_csv('/content/drive/MyDrive/dataset/iris.csv')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfle=data
dfle['Class']=le.fit_transform(dfle['Class'])
y=data['Class']
x=data.drop('Class',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.linear_model import LogisticRegression
lr=linear_model.LogisticRegression()
model=lr.fit(x_train,y_train)
predictions=model.predict(x_test)
pickle.dump(lr,open('modelpk','wb'))