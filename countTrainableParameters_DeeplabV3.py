# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:42:18 2018

@author: muhammad.asad
"""

import tensorflow as tf
import numpy as np
from deeplabv3 import *
from countParametersHelper import *
import os

modelName = "deeplabv3"

x = tf.placeholder(tf.float32, [None, 512, 512, 3])

#if(1 == 2):
net, end_points = deeplabv3(x, 21)

sess = tf.Session()
writer = tf.summary.FileWriter(os.path.join("graphs", modelName) + "graph" + os.path.sep, sess.graph)
init = tf.global_variables_initializer()

sess.run(init)	
im = np.zeros((1, 512, 512, 3))

createCSVTrainParamsFromGraph(modelName)
ep = sess.run(end_points, feed_dict={x: im})
createCSVModelParamsFromEndpoints(modelName, ep)
