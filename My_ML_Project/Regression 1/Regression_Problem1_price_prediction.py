# problem statement :
# With given dataset , try to predict the Price based on past data analysis :

# prediction of price
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('C:/Users/Anket Lale/Downloads/data.csv')
print(dataset)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

plt.plot(y,x,'o')
plt.show()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=1/3,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression	
poly_reg=PolynomialFeatures(degree=4)
x_poly= poly_reg.fit_transform(x_train) 
poly_reg.fit(x_poly,y_train)  
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y_train)
print(lin_reg_2.predict(poly_reg.fit_transform([[25.12848465]])))
