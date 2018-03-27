	
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:42:18 2018

@author: muhammad.asad
"""

import tensorflow as tf
import numpy as np

from countParametersHelper import *
import os

from model import *

tf.reset_default_graph()



modelName = "segnet"

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
batch_size = 1
phase_train = tf.placeholder(tf.bool, name='phase_train')

logits, end_points = inference_with_endpoints(x, x, batch_size, phase_train)

sess = tf.Session()
writer = tf.summary.FileWriter(os.path.join("graphs", modelName) + "graph" + os.path.sep, sess.graph)
init = tf.global_variables_initializer()

sess.run(init)	
im = np.zeros((1, 512, 512, 3))

createCSVTrainParamsFromGraph(modelName)
ep = sess.run(end_points, feed_dict={x: im, phase_train: True})
createCSVModelParamsFromEndpoints(modelName, ep)
