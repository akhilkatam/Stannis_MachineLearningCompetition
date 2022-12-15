import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics

dataset=pd.read_csv("train.csv")
print(dataset.head())

traindata=dataset.drop('soldierId',axis=1)
traindata=traindata.drop('shipId',axis=1)
traindata=traindata.drop('attackId',axis=1)
traindata=traindata.drop('bestSoldierPerc',axis=1)

trainlabel=dataset['bestSoldierPerc']
X_train, X_test, y_train, y_test = train_test_split(traindata, trainlabel, test_size=0.20, random_state=0)

svclassifier = SVR(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(svclassifier.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))