#!/usr/bin/python

from keras.models import load_model

import pandas as pd
import numpy as np

# Read data
test = pd.read_csv('test.csv')
X_test = (test.ix[:,:].values).astype('float32')

# 28x28 pixels
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

# pre-processing: divide by max and substract mean
scale = 255 
X_test /= scale

mean = np.std(X_test)
X_test -= mean

# load model
model = load_model('model.h5')
print("Loaded model from disk")

print('Predicting...')
prediction = model.predict(X_test)

#write_preds(prediction, "prediction.csv")
results = []
for i in range(0,len(prediction)):
    # prediction put into list
    tmp_result = np.argmax(prediction[i])
    results.append(tmp_result)

#results=kn.predict(tdata)
res = pd.DataFrame()
res['ImageID'] = list(range(1,len(prediction)+1))
res['Label'] = results
res.to_csv('prediction.csv',index=False)
print("prediction.csv saved to disk")
