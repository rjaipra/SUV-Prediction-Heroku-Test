# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:57:10 2019

@author: rjaipra
"""

import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#Read the Data set
df=pd.read_csv('suv_data.csv')


#converting gender to numerical value
sex=pd.get_dummies(df['Gender'],drop_first=True)
df=pd.concat([df,sex],axis=1)
df.drop('Gender',axis=1,inplace=True)
x=df.drop('Purchased',axis=1)
y=df['Purchased']

#Model creation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
logmod=LogisticRegression()
logmod.fit(X_train,y_train)
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))