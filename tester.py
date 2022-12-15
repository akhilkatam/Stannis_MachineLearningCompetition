import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import csv
import pickle

print("Reading testset")

dataset=pd.read_csv("test.csv")
traindata=dataset.drop('soldierId',axis=1)
traindata=traindata.drop('shipId',axis=1)
traindata=traindata.drop('attackId',axis=1)
ids=dataset['soldierId']

print("loading model")

rfmodel=pickle.load(open('rf30.model','rb'))

print("predicting")

result=rfmodel.predict(rfmodel)

print("writing into csv")

with open('outRF30.csv', 'w', newline="") as writeFile:
    writer = csv.writer(writeFile)
    tempo=[]
    tempo.append("soldierId")
    tempo.append("bestSoldierPerc")
    writer.writerow(tempo)
    for i in range(0,len(ids)):
        temp=[]
        temp.append(ids[i])
        temp.append(result[i])
        writer.writerow(temp)

