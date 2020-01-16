# PROBLEM STATEMENT :
# here we have 3 types of wines and their attributes
# try to predict the classification of wine based on attributes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:/Users/Anket Lale/Desktop/wine.csv')

x = dataset.iloc[:, 1: ].values
y = dataset.iloc[:, 0].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

count=0
for i in range(len(y_pred)):
	if(y_pred[i]==y_test[i]):
		count=count+1

print("ACCURACY = ",(count/len(y_pred)*100))