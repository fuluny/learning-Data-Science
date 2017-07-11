#!/usr/bin/python

import pandas as pd

from sklearn.model_selection import train_test_split as ttsp

#from sklearn.decomposition import KernelPCA as kp
from sklearn.decomposition import IncrementalPCA as kp

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#d = pd.read_csv('train.csv')
d = pd.read_csv('data.csv')

im = d.iloc[:,1:].copy(deep=False) # training and test images
lb = d.iloc[:,0].copy(deep=False) # training and test images
im[im>0]=1

images, timages, labels, tlabels = ttsp(im,lb,test_size=0.2,random_state=42)

#n = 25
batch =5000

# dimensionality reduction
print('Reducing dimensions...')
red = kp(n_components=25,batch_size=batch)
red.fit(images[:batch],labels[:batch])
print('Transforming dimensions...')
redim = red.transform(images)
redtim = red.transform(timages)

# Classifier
clf = SVC()
print('Learning...')
clf.fit(redim, labels)
print('Predicting...')
predictions = clf.predict(redtim)
print("Accruacy: %0.2f" % (accuracy_score(tlabels, predictions)))
print(classification_report(tlabels, predictions))

