#!/usr/bin/env python
# coding: utf-8

# In[22]:

import tensorflow as tf
#MiniVGGNet on cifar10
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from minivggnet import *
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import argparse
import numpy as np


#construct the argument parser for the output

ap = argparse.ArgumentParser()
ap.add_argument("-o" , "--output" , required =True, help ="path to the output loss accuracy plot")
args  = vars(ap.parse_args())
print("loading the cifar10 dataset....")

((trainX , trainY) , (testX ,  testY)) =cifar10.load_data()

trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY =lb.fit_transform(trainY)
testY =lb.transform(testY)

labelNames =["airplane" , "bird" ,  "cat" ,"dog","automobile" , "deer" , "frog", "horse" , "ship" , "truck"]

#lets compile our model and start training VGGnet

print("compiling model....")
opt = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.3, decay=0, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss= "categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
print("info training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
     batch_size=64, epochs=40, verbose=1)

predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history[ "loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label=  "val_acc")
plt.title("training loss and accuracy on cifar10")
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.legend()
plt.savefig(args["output"])

















