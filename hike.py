import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy

dataset = numpy.loadtxt("C:/Users/akhil/Desktop/train.csv", delimiter=",",skiprows=1)


dataset=pd.read_csv("C:/Users/akhil/Desktop/train.csv")
print(dataset.shape)

