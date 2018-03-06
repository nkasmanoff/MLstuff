#Noah Kasmanoff, the breast cancer dataset.

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import Sequential
from keras.layers import Dense
import tensorflow 
import pandas as pd
import numpy as np

np.random.seed(69)

FRAC = 4/5  #fraction of data split for training and testing. 
data = pd.read_csv(sys.argv[1], header = None, names = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli','Mitoses','Class'],skiprows=[23,40,139,145,158,164,235,249,275,292,294,297,315,321,411,617]) 
training = data.sample(frac = FRAC, axis = 0)
testing  = data.drop(training.index)


x_train = training.values[:,1:10]
y_train = training['Class'].values
#y_train[y_train<3] = 0 #hot encoding that wasn't correct. 
#y_train[y_train>3] = 1 
y_train = pd.get_dummies(training['Class']).values #real hot encoding. 
y_test = testing['Class'].values
#y_test[y_test<3] = 0 #hot encoding that wasn't right, so I left it out
#y_test[y_test>3] = 1 

x_test = testing.values[:,1:10]
y_test = pd.get_dummies(testing['Class']).values
del data



model=Sequential()
model.add(Dense(12,input_dim=9,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='relu'))

model.add(Dense(2,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=25,epochs=25,validation_split=.2,verbose=0)


scores = model.evaluate(x_test,y_test)
print("Testing results: " + str(scores[1]) + "% accuracy.")