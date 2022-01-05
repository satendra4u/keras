#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 01:55:04 2021

@author: satyen
"""

import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(5 * x)
print(x.shape)
print(x.dtype)


if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
  

var = tf.Variable([0.0, 0.0, 0.0])

var.assign([1, 2, 3])

var.assign_add([1, 1, 1])

print (var)


x = tf.Variable(1.0)

def f(x):
  y = x**2 + 2*x - 5
  return y

f(x)
  