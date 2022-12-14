#Import the Data
import pandas as pd
Income_Data

#Data F
Income_Data.dtypes

Income_Data.info()

#Viewing Data
Income_Data

Income_Data.head(2)

Income_Data.tail(3)

Income_Data.sample(2)

# Check column names
Income_Data.columns

# Data Description
Income_Data.describe()

Income_Data.describe(include = 'all')

# Tranpose Data 
Income_Data

Income_Data.INCOME

Income_Data=Income_Data.AREA
Income_Data

Income_Data.iloc[0]

x_Income_Data = Income_Data.drop(['INCOME','AREA'], axis=1)
x_Income_Data

y_Income_Data = Income_Data['SEX']
y_Income_Data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(x_Income_Data, y_Income_Data)
Xtrain.head()

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()                       
model.fit(Xtrain, ytrain)                 
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

from sklearn.metrics import classification_report

print(classification_report(ytest, y_model))

from sklearn.metrics import confusion_matrix 
confusion_matrix(ytest, y_model)

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
confusion_matrix = metrics.confusion_matrix(ytest, y_model)

print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=np.unique(y_mc))

cm_display.plot()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(ytest, y_model))
