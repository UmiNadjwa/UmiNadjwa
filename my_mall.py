# -*- coding: utf-8 -*-
"""Excercise:Mall_Customer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iu6By-1jSjEZFhshAFrmwQ6nyPn4hM90
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
mc = pd.read_csv('/content/mall_customer.csv')

mc.head()
mc.tail()

mc.describe()
mc.info()

x_mc = mc.drop(['Genre','CustomerID'], axis=1)  
x_mc

y_mc = mc['Genre']
y_mc

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(x_mc, y_mc)
Xtrain.head()