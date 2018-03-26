# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:42:18 2018

@author: muhammad.asad
"""

import tensorflow as tf
import numpy as np
from FCN import *
from countParametersHelper import *
import os
tf.reset_default_graph()
modelName = "FCN"

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

#if(1 == 2):
net, _, end_points = inference(x, keep_probability)

sess = tf.Session()
writer = tf.summary.FileWriter(os.path.join("graphs", modelName) + "graph" + os.path.sep, sess.graph)

init = tf.global_variables_initializer()

sess.run(init)	
im = np.zeros((1, 512, 512, 3))
ft = 1.0

createCSVTrainParamsFromGraph(modelName)
ep = sess.run(end_points, feed_dict={x: im, keep_probability: ft})
createCSVModelParamsFromEndpoints(modelName, ep)
