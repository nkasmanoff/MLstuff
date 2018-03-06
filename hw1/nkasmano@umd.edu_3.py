#Noah Kasmanoff
#MNIST classification set. 

#Import packages, make sure warnings are muted.
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')


from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten 
from keras.layers.core import Dense
from keras.optimizers import SGD,RMSprop,Adam
import numpy as np 
from keras.utils import np_utils  #hot encodes

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#set random seed for replication purposes
np.random.seed(69)


#using a class of CNN's known as LeNet, known for being 
#particularly good at classifying MNIST images. 
#Source: https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

class LeNet:
    @staticmethod
    def build(input_shape,classes):
        model = Sequential()
        model.add(Conv2D(20,kernel_size=5,padding="same",input_shape=input_shape,data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(50,kernel_size=5,border_mode="same",data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

#import training data, convert to images rather than arrays. 
# UPDATE 3/5: I also have to split these arrays up into training and testing. I will do this using 
#sklearn train_test_split, but to assure that even though this is done randomly I will assign images and labels the same seed so it lines up. 
images = pd.read_csv(sys.argv[1])

train_images , test_images = train_test_split(images,random_state=4)
labels = pd.read_csv(sys.argv[2])
test_labels, test_labels = train_test_split(labels,random_state=4)
xtrain = train_images.values
Xtrain = xtrain.reshape(xtrain.shape[0],1,28,28).astype('float32')
#OHE
Ytrain = np_utils.to_categorical(train_labels.values)
#now the test values
xtest = test_images.values
Xtest = xtest.reshape(xtest.shape[0],1,28,28).astype('float32')
#OHE
Ytest = np_utils.to_categorical(test_labels.values)

del train_images,train_labels  #data cleanup. 

model = LeNet.build((1,28,28),10)

#build and compile NN.
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

history = model.fit(Xtrain,Ytrain,batch_size=128,epochs=2,validation_split=0.2,verbose=0)
#Here is a print statement 

scores = model.evaluate(X_test,Y_test)
print("The accuracy of this CNN on the testing set is " + str(100*scores[1]) + "%.")