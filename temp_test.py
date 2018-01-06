# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:19:48 2018

@author: Avi
"""
from network_class import *
import tensorflow as tf

sess = tf.Session()
btach = tf.Variable(tf.truncated_normal([64, 1, 66650, 1], stddev=0.1))
net = DSRNet(btach, 50, True)