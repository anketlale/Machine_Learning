# import 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('C:/Users/Anket Lale/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

print(dataset)

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

print(x.shape)
print(y.shape)
y=y.values.reshape(50,1)
print(y.shape)

#encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)
# avoid dummy variable trap
x=x[:,1:]


from sklearn.model_selection import train_test_split
x_train ,x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


print("Actual Values := ")
print(y_test)
print("Predicted values := ")
print(y_pred)

