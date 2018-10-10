# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:41:40 2018

@author: rajneesh.jha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv(r"C:\Users\rajneesh.jha\Downloads\kaggle_mnist\train.csv")
test = pd.read_csv(r"C:\Users\rajneesh.jha\Downloads\kaggle_mnist\test.csv")

print(train.shape)
print(test.shape)


X_train=train.iloc[:,1:].values.astype('float32')
test=test.values.astype('float32')
Y_train=train.iloc[:,0].values.astype('int')
Y_train=train['label']

Y_train.value_counts(dropna=False)

img_rows, img_cols = 28, 28
num_classes = 10  # digits 0 - 9




X = X_train.reshape(X_train.shape[0], img_rows, img_cols)
test_fin = test.reshape(test.shape[0], img_rows, img_cols)

print(X.shape)
# add another dimension for color channel
X = X.reshape(X.shape[0], img_rows, img_cols, 1)
test_fin = test_fin.reshape(test_fin.shape[0], img_rows, img_cols, 1)

print(X.shape)
print(test_fin.shape)

X = X / 255
test_fin = test_fin / 255
from keras.utils import to_categorical
Y_train = to_categorical(Y_train)


from sklearn.model_selection import train_test_split
X_tr,X_valid,Y_tr,Y_valid=train_test_split(X, Y_train, test_size=0.1, random_state=42)

from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, MaxPool2D
from keras.models import Sequential,Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import keras
from keras import backend as K


model=Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 # pads the "frames" of the image with 0's, so that convolution reaches the edges
                 padding='same',
                 input_shape=(28,28,1)))


model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(Dropout(0.2))

model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Conv2D(64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))



# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




history=model.fit(X, Y_train, 
                    epochs=5,
                    validation_data=(X_valid, Y_valid))


# make predictions
preds = model.predict_classes(test_fin)


submission = pd.DataFrame(
    {'ImageId': list(range(1, len(preds)+1)), 'Label': preds})

submission.to_csv(r'C:\Users\rajneesh.jha\Desktop\submission_new1.csv', index=False)























































