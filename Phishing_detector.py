import pandas as pd   #--->data manupulation and data anaysis in python, pd is just short name intead of using full python
# Load the data into variable data. read from "phishing.csv "and load to 'data'
data = pd.read_csv('phishing.csv')

print(data.head())   #-->print first 5 rows of data set
print(data.isnull().sum())   # check if data isn't have any null value(return true if have) and count how many time data have null value.

x = data.drop(['class'], axis=1)  #-->drop the class where results are contain # Features
y = data['class']     # class sata data store in y # Target

#X = data.drop(['Result'], axis=1)  # Features
#y = data['Result']                 # Target

#print(data.columns)   # print the name of the columns contain by data
from sklearn.model_selection import train_test_split   #--> module  in sklearn thats helps to split the data, and rnadomely divide data into train and test data 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)   # here test is 30% and train is 70 % random state is 42 random no
#give you same results each time you run the code

from sklearn.ensemble import RandomForestClassifier  #Random Forest is an ensemble machine learning algorithm that builds multiple decision trees and
#combines their outputs to improve accuracy and reduce overfitting.
from sklearn.metrics import accuracy_score, classification_report #These are tools to evaluate how well your model is performing.

model = RandomForestClassifier()
model.fit(x_train, y_train)   #--> Train the model using traing data 


y_pred = model.predict(x_test)  # Predict the test data 


print("Accuracy: ", accuracy_score(y_test, y_pred)) # --> give accuracy report 
print(classification_report(y_test, y_pred))   # give classification report

#Plotation
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model,x_test,y_test)
plt.show()

import joblib
joblib.dump(model, 'phishing_model.pkl')