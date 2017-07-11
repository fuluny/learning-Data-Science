# #!/usr/bin/python

import os
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers

from matplotlib import pyplot as plt

print('Loading data...')
data = pd.read_csv('fer2013.csv')
#data = pd.read_csv('testdata.csv')
im = data['pixels']
im_list = []

print('Pre-processing data...')
for i in range(len(im)):
	im_list.append(list(map(int,im[i].split())))

X_train = np.asarray(im_list).astype('float32')
y_train = np_utils.to_categorical(np.asarray(data['emotion']))

X_train *= 2.0/255
X_train -= 1

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Parameters were chosen from most commonly used and sometimes at random
# Further development of the model may be needed
print('Making model')
model = Sequential()
# Dense define number of nodes
model.add(Dense(1000, input_dim=input_dim))
# Activation defines the output
model.add(Activation('relu'))
# Dropout to avoid overfitting.
model.add(Dropout(0.15))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
print(model.summary())

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

print("Training...")
model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=2)

scores = model.evaluate(X_train, y_train, verbose=0)
print(scores)

# save model to HDF5
model.save('model.h5')
print("Saved model to disk")

