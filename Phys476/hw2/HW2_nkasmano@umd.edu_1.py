#Noah Kasmanoff, HW2 Problem 1 Phys476 
#Classify mice using k-nearest-neighbors (KNN)
#http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression


#import the necessary packages

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

data = pd.read_csv(sys.argv[1])

#Remove columns that obviously don't play a role. 
del data['MouseID']


#Make the output y a separate array. 
y = data['class']
y = pd.get_dummies(y).values
del data['class']

#assign missing values as outlier values, and hot encode and create input data. 
data = data.fillna(-9999)

#Now hot-encode, plus obtain array of X (input) vals. 
X = pd.get_dummies(data).values

#separate into training and testing values randomly. 
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#Create model object via K neighbors, and train it. 
clf = neighbors.KNeighborsClassifier()

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)