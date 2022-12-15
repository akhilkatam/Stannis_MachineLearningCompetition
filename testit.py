import tensorflow as tf
import numpy

print("Loading data for testing...")
dataset = numpy.loadtxt("train.csv", delimiter=",",skiprows=1)
print("Preparing dataset...")
dataset=dataset[numpy.random.randint(dataset.shape[0], size=10), :]
testinputs=dataset[0:,3:24]
testoutputs=dataset[0:,24]
print("Loading model")
model=tf.keras.models.load_model("seed7.model")
print("testing")
score=model.evaluate(testinputs, testoutputs)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))