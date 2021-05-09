# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:26:28 2021

@author: cihan
"""


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,mean_absolute_error,mean_squared_error


os.chdir('G:\Okul\Bitirme2')
df = pd.read_csv(r'afg.csv')
""" 
vaka kurtulan sayısını X verisetimiz için , ölüm sayısını da tahmin
edeceğimizden dolayı y dizimize ekleyelim
"""

"""
grafik
"""
corelation = df.corr()
plt.subplots(figsize=(20,15))
a = sns.heatmap(corelation,annot=True)

#x= np.array(df.drop(['deaths'],1))
#y= np.array(df[output])

X = df.iloc[:, [3,5]].values
y = df.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)




#SVC
svc=SVC(kernel='linear', random_state = 0)


svc.fit(x_train,y_train)

y_pred3=svc.predict(x_test)

accuracy_score(y_test,y_pred3)


print(f"Test Set Accuracy : {accuracy_score(y_test,y_pred3) * 100} %\n\n")

print(f"Classification Report : \n\n{classification_report(y_test, y_pred3)}")





