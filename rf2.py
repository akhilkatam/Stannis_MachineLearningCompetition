import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import csv

dataset = pd.read_csv('train.csv')

trainingdata=dataset.drop('soldierId',axis=1)
trainingdata=trainingdata.drop('shipId',axis=1)
trainingdata=trainingdata.drop('attackId',axis=1)
trainingdata=trainingdata.drop('bestSoldierPerc',axis=1)

traininglabel=dataset['bestSoldierPerc']

regressor = RandomForestRegressor(n_estimators=33, random_state=0)
regressor.fit(trainingdata, traininglabel)

testset=pd.read_csv('test.csv')
X_test=testset.drop('temporaryhead',axis=1)
X_test=X_test.drop('index',axis=1)
X_test=X_test.drop('soldierId',axis=1)
X_test=X_test.drop('shipId',axis=1)
X_test=X_test.drop('attackId',axis=1)

ids=testset['soldierId']

y_pred = regressor.predict(X_test)

with open('RF30.csv', 'w', newline="") as writeFile:
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

# print(regressor.score(X_test, y_test))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(y_test, y_pred)))