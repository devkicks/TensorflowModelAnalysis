# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:42:18 2018

@author: muhammad.asad
"""

import tensorflow as tf

from deeplabv3 import *
from countParametersHelper import *

x = tf.placeholder(tf.float32, [None, 512, 512, 3])

#if(1 == 2):
net, end_points = deeplabv3(x, 21)

sess = tf.Session()
writer = tf.summary.FileWriter("deeplabv3graph/", sess.graph)
init = tf.global_variables_initializer()

sess.run(init)	
#total_parameters = 0
##i = 1
#for variable in tf.trainable_variables():
#    # shape is an array of tf.Dimension
#    shape = variable.get_shape()
#    
#    print(variable.name + ': ' + str(shape) )
##    print(len(shape))
#    variable_parameters = 1
#    for dim in shape:
#        print(dim)
#        variable_parameters *= dim.value
#    print(variable_parameters)
#    total_parameters += variable_parameters
#    
##    i+=1
#    
#print(total_parameters)
#

createCSVTrainParamsFromGraph("DeeplabV3")