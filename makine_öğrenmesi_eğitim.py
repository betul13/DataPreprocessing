# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:38:36 2023

@author: betul
"""
import pandas as pd
import numpy as np
import matplotlib as plt

dataset = pd.read_csv(r"C:\Users\betul\Desktop\machine_learning\Machine Learning A-Z (Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv", encoding= 'unicode_escape')

x = dataset.iloc[:,:-1].values#bağımsız değerleri aldık
y = dataset.iloc[:,-1].values #sadece son sütun bağımlı 
print(x)
print(y)

from sklearn.impute import SimpleImputer #Boş olan sayısal kısımları doldurmak için kullanulır
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") #boş kısımları ortalamayla doldurur.
x[:,1:3] = imputer.fit_transform(x[:,1:3])#fonksiyonla eşleme ve transform
#x[:,1:3] = imputer.transform(x[:,1:3]) # boş olan yerler ortalamayla değiştirildi.
print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough") 
x = np.array(ct.fit_transform(x))
print(x)
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
y = Le.fit_transform(y)
print(y)

#eğitim ve test seti ayırma
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state=(1))

#Özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:]= sc.fit_transform(x_train[:,3:])
