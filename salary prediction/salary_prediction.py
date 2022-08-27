import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
data_set=pd.read_csv("Salary_Data.csv")
data_set.head()
x=data_set.drop(['Salary'],axis=1)
y=data_set.drop(['YearsExperience'],axis=1)
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)
print(y_train)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
m=lr.fit(x_train,y_train)
pickle.dump(lr,open('model (1).pkl','wb'))