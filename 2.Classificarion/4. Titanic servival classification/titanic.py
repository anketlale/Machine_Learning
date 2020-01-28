#!/usr/bin/env python
# coding: utf-8

# In[240]:


from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[241]:


train = pd.read_csv('C:/Users/Anket Lale/Desktop/titanic/train.csv')
test = pd.read_csv('C:/Users/Anket Lale/Desktop/titanic/test.csv')


# In[ ]:





# # data analysis

# In[ ]:





# In[242]:


# training


# In[243]:


train.head(5)


# In[244]:


train.info()


# In[245]:


train.shape


# In[ ]:





# In[246]:


test.head(5)


# In[247]:


test.info()


# In[248]:


test.shape


# In[ ]:





# In[249]:


# we can see that for train - 
# age is many missing values
# similarly cabin 
# similarly embarked


# In[250]:


test.isnull().sum()


# In[251]:


test.isnull().sum()


# In[ ]:





# In[ ]:





# # Visualization 

# In[252]:


# setting some defaults

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[253]:


# skipped


# In[254]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:





# In[255]:


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")


# In[ ]:





# In[ ]:





# # start converting text value to number

# # 1.name

# In[256]:


# name is not useful but gender via name is useful
# there are types in name like mr , mrs , miss , other != gender
# hence make it value


# In[257]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[258]:


train.info()


# In[259]:


test.info()


# In[ ]:





# In[260]:


train['Title'].value_counts()


# In[ ]:





# In[261]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:





# In[262]:


train


# In[ ]:





# In[263]:



# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[264]:


train.head(5)


# In[265]:


test.head()


# In[ ]:





# # 2.sex

# In[266]:


sex_mapping = {"male" : 0 , "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[267]:


train.head(5)


# In[ ]:





# In[268]:


test.head(5)


# In[ ]:





# # Handling missing values

# # 1. Age 

# In[ ]:





# In[269]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)


# In[270]:


train["Age"].isnull().sum()


# In[ ]:





# In[271]:


test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].isnull().sum()


# In[ ]:





# In[272]:


train.head(5)


# In[ ]:





# In[273]:


train['Age'].value_counts()


# In[274]:


# make age as catogorical


# In[275]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[ ]:





# In[276]:


train.head()


# In[ ]:





# In[277]:


train["Age"].value_counts()


# In[278]:


# it worked !!!


# In[279]:


train.isnull().sum()


# # 2.Embarked

# In[280]:


train['Embarked'].isnull().sum()


# In[281]:


train["Embarked"].value_counts()


# In[282]:


# here Embarked has most S value hence replace 2 missing value with S


# In[283]:


for i in train_test_data:
    i['Embarked']=i['Embarked'].fillna('S')


# In[ ]:





# In[284]:


train['Embarked'].isnull().sum()


# In[285]:


train.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[286]:


train['Embarked'].value_counts()


# In[287]:


# Embarked - converting into numeric value


# In[288]:


embarked_mapping = {"S": 0 , "C":1 , "Q":2} 


# In[289]:


for i in train_test_data:
    i['Embarked']=i['Embarked'].map(embarked_mapping)


# In[290]:


train['Embarked'].value_counts()


# In[291]:


# done


# In[ ]:





# In[292]:


train.isnull().sum()


# In[ ]:





# # Fare 

# In[293]:


# in train - we does not have blank value
# but for test , we need to handle


# In[294]:


test['Fare'].isnull().sum()


# In[295]:


# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[296]:


train.isnull().sum()


# In[ ]:





# In[ ]:





# In[297]:


train.head()


# In[298]:


max(train['Fare'])


# In[299]:


min(train['Fare'])


# In[300]:


# now group to Fare-


# In[301]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[302]:


train.head()


# In[ ]:





# In[303]:


train['Fare'].value_counts()


# In[ ]:





# # Cabin

# In[304]:


train['Cabin'].value_counts()


# In[305]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[306]:


train['Cabin'].value_counts()


# In[307]:


# grouping - to make it numeric


# In[308]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[309]:


train['Cabin'].value_counts()


# In[ ]:





# In[ ]:





# In[310]:


# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[311]:


train['Cabin'].isnull().sum()


# In[ ]:





# # Make Family column to remove SibSp and Parch

# In[312]:


train.head()


# In[313]:


for i in train_test_data:
    i['Family']=i['SibSp']+i['Parch']+1


# In[314]:


test.head()


# In[ ]:





# In[315]:


test['Family'].value_counts()


# In[316]:


family_map={1:0 , 2:0.4 , 3:0.8 , 4:1.2 , 5:1.6 , 6:2 , 7:2.4 , 8:2.8 ,9:3.2 , 10:3.6 , 11:4}

for i in train_test_data:
    i['Family']=i['Family'].map(family_map)


# In[317]:


train.head()


# In[318]:


test.head()


# In[319]:


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[320]:


train.head()


# In[ ]:





# # make training and testing data ready

# In[321]:


train_data = train.drop('Survived',axis=1)


# In[322]:


target_data= train['Survived']


# In[323]:


train_data.shape


# In[324]:


target_data.shape


# In[ ]:





# # model

# # Testing

# In[325]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[326]:


# Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:





# In[327]:


# 1. KNN


# In[328]:


kn = KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)


# In[329]:


#common for all algo
scoring = 'accuracy'
score = cross_val_score(kn, train_data, target_data, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)


# In[ ]:





# In[330]:


# 2.SVM

sv = SVC()
#common for all algo
scoring = 'accuracy'
score = cross_val_score(sv, train_data, target_data, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)


# In[ ]:





# In[331]:


# 3.Naive bayes
nb = GaussianNB()
#common for all algo
scoring = 'accuracy'
score = cross_val_score(nb, train_data, target_data, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)


# In[ ]:





# In[332]:


# 4.Decision tree 
dt = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
scoring = 'accuracy'
score = cross_val_score(dt, train_data, target_data, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)


# In[ ]:





# In[333]:


# 5.random forest
rf = RandomForestClassifier(n_estimators = 16 , criterion = 'entropy',random_state = 0 )
scoring = 'accuracy'
score = cross_val_score(rf, train_data, target_data, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)


# In[ ]:





# In[ ]:





# # TESTING 

# In[ ]:





# In[ ]:





# In[334]:


test.isnull().sum()


# In[335]:


test.head(10)


# In[336]:


train.head()


# In[337]:


test_data = test.drop('PassengerId',axis=1).copy()


# In[ ]:





# In[ ]:





# In[ ]:





# In[344]:


output = test


# In[352]:


dd = ['Pclass','Sex','Age','Fare','Cabin','Embarked','Title','Family']


# In[ ]:





# In[357]:


output=output.drop(dd,axis=1).copy()


# In[339]:


test_data.info()


# In[ ]:





# In[341]:


sv.fit(train_data , target_data)
y_pred = sv.predict(test_data)


# In[ ]:





# In[358]:


output['Survived']=y_pred


# In[359]:


output


# In[ ]:





# In[ ]:





# # with accuracy : 83.5
