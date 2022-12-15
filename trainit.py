import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy


numpy.random.seed(7)
print("Loading dataset for training.......")
dataset = numpy.loadtxt("train.csv", delimiter=",",skiprows=1)
print("Dataset loaded.\n")
print("Preparing data for training......")
traininputs=dataset[0:1000000,3:24]
trainoutputs=dataset[0:1000000,24]
print("Data prepared.")
print("Creating model......")
model=Sequential()
model.add(Dense(32,input_dim=21,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model created.")

print("Training data.....")
model.fit(traininputs, trainoutputs, epochs=10, batch_size=64)
print("Data trained, saving the model.")
model.save("seed7.model")
print("Done")
