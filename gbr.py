import sklearn
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import csv
#4357336,25


dataset=pd.read_csv("train.csv")
print(dataset.head())

traindata=dataset.drop('soldierId',axis=1)
traindata=traindata.drop('shipId',axis=1)
traindata=traindata.drop('attackId',axis=1)
traindata=traindata.drop('bestSoldierPerc',axis=1)

trainlabel=dataset['bestSoldierPerc']
X_train, X_test, y_train, y_test = train_test_split(traindata, trainlabel, test_size=0.20, random_state=0)

regressor = GradientBoostingRegressor(n_estimators=25, random_state=0)
regressor.fit(X_train, y_train)

dataset=pd.read_csv("test.csv")
print(dataset.head())

traindata=dataset.drop('soldierId',axis=1)
traindata=traindata.drop('shipId',axis=1)
traindata=traindata.drop('attackId',axis=1)
ids=dataset['soldierId']
print(len(ids))

y_pred = regressor.predict(X_test)
print(len(y_pred))
with open('outputGBC25.csv', 'w', newline="") as writeFile:
    writer = csv.writer(writeFile)
    tempo=[]
    tempo.append("soldierId")
    tempo.append("bestSoldierPerc")
    writer.writerow(tempo)
    for i in range(0,len(ids)):
        temp=[]
        temp.append(ids[i])
        temp.append(y_pred[i])
        writer.writerow(temp)
print(y_pred)
#
# print(regressor.score(X_test, y_test))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))