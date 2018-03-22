#Noah Kasmanoff 
#Predict cases of heart disease using all 3 of the following techniques:
#random forest method. 
#http://archive.ics.uci.edu/ml/datasets/Heart+Disease
# (Links to an external site.)***Use the processed.hungarian.data set for all cases, be sure to check for missing data and handle it accordingly. (changed)



import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
np.random.seed(4)

#Imputer used to replace missing values whith the median 

NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
data = pd.read_csv(sys.argv[1],header=None,names=NAMES)
data.replace(['?'],np.nan,inplace=True)
#ok almost done now. 
data = pd.DataFrame(Imputer(missing_values='NaN',strategy='median',axis=0).fit_transform(data),columns=NAMES)

y = data['num']
#y = pd.get_dummies(y).values don't need to hot encode these since they're already classified. 
del data['num']

X = data.values
del data

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.22)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=2, random_state=0) #can play with attributes.
rf.fit(X_train,y_train)
score3 = rf.score(X_test,y_test)

print(score3)


