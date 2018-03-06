#Noah Kasmanoff

#Classify the species of Iris given in the image. Use the bezdekIris.data file.

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

np.random.seed(69)

FRAC = 6/9

# load dataset
data = pd.read_csv(sys.argv[1], header=None)
#dataset = dataframe.values
training = data.sample(frac = FRAC, axis = 0)
testing  = data.drop(training.index)

X_train = training.values[:,0:4]
Y_train = pd.get_dummies(training[4]).values

X_test = testing.values[:,0:4]
Y_test = pd.get_dummies(testing[4]).values
del training,testing
#Create the ANN. Already get a very strong result with this one. 
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
#note softmax is used for classification problems as the final entry before the output. 
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#need to use a large number of epochs for this set, still end up with non overfit data based on the validation reisutls plus model evaluation. 
model.fit(X_train,Y_train,batch_size=35,epochs=200,validation_split=0.2,verbose=0)



scores = model.evaluate(X_test,Y_test)
print("The accuracy of this ANN on the testing set is " + str(100*scores[1]) + "%.")