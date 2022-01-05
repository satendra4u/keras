#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:03:40 2022

@author: satyen
"""

#from keras.models import Sequential 
#from keras.layers import Dense, Activation , Dropout 

#model = Sequential()
 
#model.add(Dense(512, activation = 'relu', input_shape = (784,)))
#model.add(Dropout(0.2))
#model.add(Dense(512, activation = 'relu')) 
#model.add(Dropout(0.2)) 
#model.add(Dense(512, activation='softmax'))
#model.summary()


from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 
from keras import regularizers
from keras import constraints


model = Sequential() 

model.add(Dense(32, input_shape=(16,), kernel_initializer = 'he_uniform', 
   kernel_regularizer = None, kernel_constraint = 'MaxNorm', activation = 'relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(8)) 
model.summary()


### variance scalling -------

##
##Generates value based on the input shape and output shape of the layer along with the specified scale.

#from keras.model import Sequential
#from keras.layers import Activation, Dense
#from keras import initializers

