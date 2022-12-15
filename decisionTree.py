import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
#4357336,25


dataset=pd.read_csv("train.csv")
print(dataset.head())

traindata=dataset.drop('soldierId',axis=1)
traindata=traindata.drop('shipId',axis=1)
traindata=traindata.drop('attackId',axis=1)
traindata=traindata.drop('bestSoldierPerc',axis=1)

trainlabel=dataset['bestSoldierPerc']
X_train, X_test, y_train, y_test = train_test_split(traindata, trainlabel, test_size=0.20)

classifier = DecisionTreeRegressor()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classifier.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))