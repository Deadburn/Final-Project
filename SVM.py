# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:26:28 2021

@author: cihan
"""


import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn import metrics

#Verisetinin okunması

df = pd.read_csv(r'covid19_by_country.csv')

#Verisetimizde tahminimizi ülkelere göre yapmamız gerekiyor bu yüzden kullanıcıdan
#Ülke kodu girdisi alıp buna göre verilerimizi tek ülke biçiminde filtreleriz

dateCols = ['CountryAlpha3Code']

ulke = input('Tahmin etmek istediğiniz ülke kodunu giriniz: ')

df = pd.read_csv("covid19_by_country.csv", parse_dates=dateCols)

data=df[(df['CountryAlpha3Code']  ==  ulke)]

# verisetimizdeki null veya değeri verilmemiş değişkenler ihmal edilmelidir.

data = data.dropna() 

# verisetimizi 2 diziye bölme işlemi:
# eğer type hatası verirse Y ve y'nin sonuna .astype(float) ekleyin 

X = data.drop(['Country','CountryAlpha3Code','deaths','Date','deaths_PopPct',
               'confirmed_PopPct','DaysSince100Cases','DaysSince1Cases',
               'GRTStringencyIndex','recoveries_PopPct','recoveries_inc',
               'deaths_inc','confirmed_inc'], axis=1)
y = data[['deaths']]



x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.25)


svc = SVC()

svc.fit(x_train,y_train)


y_pred = svc.predict(x_test)


accuracy_score(y_test, y_pred)


print(f"Test Set Accuracy : {accuracy_score(y_test,y_pred) * 100} %\n\n")
print(f"Classification Report : \n\n {classification_report(y_test, y_pred)}")
print("R2 score\n")
print(metrics.r2_score(exp, y_pred))
print("Mean Squared Log Error\n")
print(metrics.mean_squared_log_error(exp, y_pred))
print("Explained Variance Score\n")
print(metrics.explained_variance_score(y_pred,y_test))
print("Mean Absolute Error\n")
print(metrics.mean_absolute_error(exp, y_pred))

## Örnek olarak AUS kodunu girdiğimizde 
## Test Set Accuracy : 39.83050847457627 %

## R2 score
## 0.9697156186323612

## Mean Squared Log Error
## 0.35953020745484654

## Explained Variance Score
## 0.9720400269443831

## Mean Absolute Error
## 24.0


