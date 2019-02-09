# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('TrainsetTugas1ML.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7]].values
y = dataset.iloc[:, 8].values
datates = pd.read_csv('TestsetTugas1ML.csv')

datatesAge = datates.iloc[:, 1].values

#Encoding categorical data , change text to number
from sklearn.preprocessing import LabelEncoder
labelencoder7 = LabelEncoder()
X[:, 0] = labelencoder7.fit_transform(X[:, 0])
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
labelencoder2 = LabelEncoder()
X[:, 2] = labelencoder2.fit_transform(X[:, 2])
labelencoder3 = LabelEncoder()
X[:, 3] = labelencoder3.fit_transform(X[:, 3])
labelencoder4 = LabelEncoder()
X[:, 4] = labelencoder4.fit_transform(X[:, 4])
labelencoder5 = LabelEncoder()
X[:, 5] = labelencoder5.fit_transform(X[:, 5])
labelencoder6 = LabelEncoder()
X[:, 6] = labelencoder6.fit_transform(X[:, 6])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

datates = datates.iloc[: , [1,2,3,4,5,6,7]].values
labelencoder7 = LabelEncoder()
datates[:, 0] = labelencoder7.fit_transform(datates[:, 0])
labelencoder = LabelEncoder()
datates[:, 1] = labelencoder.fit_transform(datates[:, 1])
labelencoder2 = LabelEncoder()
datates[:, 2] = labelencoder2.fit_transform(datates[:, 2])
labelencoder3 = LabelEncoder()
datates[:, 3] = labelencoder3.fit_transform(datates[:, 3])
labelencoder4 = LabelEncoder()
datates[:, 4] = labelencoder4.fit_transform(datates[:, 4])
labelencoder5 = LabelEncoder()
datates[:, 5] = labelencoder5.fit_transform(datates[:, 5])
labelencoder6 = LabelEncoder()
datates[:, 6] = labelencoder6.fit_transform(datates[:, 6])

# Predicting the Test set results
y_pred = classifier.predict(datates)

# See how much people who spending <=50K 
youngGreater = 0
youngLess = 0
adultGreater = 0
adultLess = 0 
oldGreater = 0
adultLess = 0
oldLess = 0
z = 0
for item in range (len(datatesAge)):
    if datatesAge[item] == 'young' and y_pred[item] == '>50K' :
        youngGreater += 1
    elif datatesAge[item] == 'young' and y_pred[item] == '<=50K' :
        youngLess += 1
    elif datatesAge[item] == 'adult' and y_pred[item] == '>50K' :
        adultGreater += 1
    elif datatesAge[item] == 'adult' and y_pred[item] == '<=50K' :
        adultLess += 1
    elif datatesAge[item] == 'old' and y_pred[item] == '>50K' :
        oldGreater += 1
    elif datatesAge[item] == 'old' and y_pred[item] == '<=50K' :
        oldLess += 1

labels = 'young', 'adult', 'old'
slices = [youngGreater, adultGreater, oldGreater] # ini ganti sama young dan >50k
cols = ['blue','red','pink']
plt.pie(slices, labels=labels, colors = cols, shadow = True, 
        explode = (0.1,0,0), autopct='%1.1f%%') 
# we explode in slices 1
# autopct to add percentage on pie
plt.show()