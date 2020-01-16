# PROLEM STATEMENT :
# in this example , we have iris dataset 
# having multiple features which is use to classify the types of iris 

# here we tried to predict the type of iris


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:/Users/Anket Lale/Desktop/iris.csv')

x = dataset.iloc[:, :4 ].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8,metric='minkowski',p=2)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

