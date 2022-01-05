#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 08:55:08 2022

@author: satyen
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints



model = Sequential()

model.add(Dense(32,input_shape=(16,),kernel_initializer='he_uniform',kernel_regularizer=None, kernel_constraint='MaxNorm', activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.summary()
