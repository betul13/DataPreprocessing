# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:22:53 2023

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

#eğitim ve test seti ayırma
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state=(1))
