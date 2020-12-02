#!/usr/bin/env python
# coding: utf-8

# ### Diabetes Prediction Using Classification Algorithm

#Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading diabetes dataset
data = pd.read_csv("Diabetes.csv")

print(data.shape)

print(data.head())

#Let's check missing values in dataset
print(data.isnull().values.any())
###### We can see that there is no missing values in dataset
print(data.info())

#Correlation of Dataset
corr = data.corr()
#plt.figure(figsize = (10,8))
#sns.heatmap(corr, annot = True)

#data.corr()

#data.corr().index

##Now, splitting dataset into independent and dependent variables
X = data.iloc[:,:-1]
y = data.iloc[:, -1]

print(X.shape, y.shape)

print(X.head())
print(y.head())

print(data['Outcome'].value_counts(normalize = True))

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 42) 

print(x_train.shape, x_test.shape)

print(y_train.shape)

print(data.rename({'DiabetesPedigreeFunction':"DPF"}, axis = 1, inplace = True))

#We check the how many zeros values present in dataset
print("Total number of rows: {}".format(len(data)))
print("Number of zero values in Glucose: {}".format(len(data.loc[data["Glucose"] == 0])))
print("Number of zero values in Pregnancies: {}".format(len(data.loc[data["Pregnancies"] == 0])))
print("Number of zero values in BloodPressure: {}".format(len(data.loc[data["BloodPressure"] == 0])))
print("Number of zero values in SkinThickness: {}".format(len(data.loc[data["SkinThickness"] == 0])))
print("Number of zero values in Insulin: {}".format(len(data.loc[data["Insulin"] == 0])))
print("Number of zero values in BMI: {}".format(len(data.loc[data["BMI"] == 0])))
print("Number of zero values in DiabetesPedigreeFunction: {}".format(len(data.loc[data["DPF"] == 0])))
print("Number of zero values in Age: {}".format(len(data.loc[data["Age"] == 0])))

#We replace these zeros values with mean values of columns
from sklearn.impute import SimpleImputer
fill_zero_values = SimpleImputer(missing_values= 0 , strategy= 'mean')

x_train = fill_zero_values.fit_transform(x_train)
x_test = fill_zero_values.fit_transform(x_test)

print(x_train.shape, x_test.shape)

#Now we import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

print(prediction)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(y_test, prediction)
print("Accuracy: {}".format(accuracy))

cm = confusion_matrix(y_test, prediction)
print("Confusion matrix: \n{} ".format(cm))

print("Classification report:\n {}".format(classification_report(y_test, prediction)))

##### Prediction on new data

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    preg = int(Pregnancies)
    gluco = float(Glucose)
    bp = float(BloodPressure)
    st = float(SkinThickness)
    insulin = float(Insulin)
    bmi = float(BMI)
    dpf = float(DPF)
    age = int(Age)
    x = [[preg, gluco, bp, st, insulin, bmi, dpf, age]]
    
    return classifier.predict(x)

##Prediction 1:

prediction = predict_diabetes(2,81,72,15,76, 30.1, 0.547, 25)
if prediction:
    print("Ooops! You have diabetes")
else:
    print("Great! You don't have diabetes.")

##Prediction 2:

prediction = predict_diabetes(1, 117, 88, 24, 145, 34.5, 0.423, 40)
if prediction:
    print("Ooops! You have diabetes")
else:
    print("Great! You don't have diabetes.")

##Prediction 3:

prediction = predict_diabetes(5, 120, 92, 10, 81, 26.1, 0.551, 67)
if prediction:
    print("Ooops! You have diabetes")
else:
    print("Great! You don't have diabetes.")

#import joblib

#joblib.dump(classifier, "diabetes_model.pkl")

