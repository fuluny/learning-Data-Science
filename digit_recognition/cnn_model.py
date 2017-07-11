#!/usr/bin/python

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.models import load_model

import os
import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('data.csv')

labels = train.ix[:,0].values.astype('int32')
X_train = train.ix[:,1:].values.astype('float32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels) 

# 28x28 pixels
in_shape = (28, 28,1)
# reshape data for Conv2D evaluation
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

# pre-processing: divide by max and substract mean
scale = 255 # 255 is max value for pixel
X_train /= scale

mean = np.std(X_train)
X_train -= mean

#in_dim = X_tdrain.shape[1]
n_out = y_train.shape[1]

model = Sequential()
# parameters need some evaluation and adjusting
model.add(Conv2D(32, kernel_size=(4, 4),
    data_format='channels_last', input_shape=in_shape))
model.add(Activation('relu'))
model.add(Dropout(0.42))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.42))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.42))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.42))
model.add(Dense(n_out))
model.add(Activation('softmax'))
print(model.summary())

# default value for optimizer, evaluate and adjust after some understanding
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# compile model
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

print("Training...")
model.fit(X_train, y_train, epochs=70, validation_split=0.1, verbose=2)

# should use test set for evaluation. fix this sometime soon
scores = model.evaluate(X_train, y_train, verbose=0)
print(scores)

# save model to HDF5
model.save('model.h5')
print("Saved model to disk")

