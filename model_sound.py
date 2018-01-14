# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
import time
import numpy as np
from network_class import *
from utils.sound_reader_kcross_val import SoundReaderKCrossValidation
from utils.write_to_log import write_log

class Model(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        
	# train
    def train(self):
        self.train_setup()

        self.sess.run(tf.global_variables_initializer())

        #Load the pre-trained model if provided
        if self.conf.pretrain_file is not '':
            self.load(self.loader, self.conf.pretrain_file)

        train_offset = np.zeros(5,dtype=np.int32)
        #for step in range(self.conf.num_steps + 1):
        for step in range(5):
            start_time = time.time()
            for fold_index in range(self.conf.k_cross_val):
                (train_input, train_labels), (cv_input, cv_labels) = self.reader.split_according_to_fold(fold_index)

                train_offset[fold_index] = (step * self.conf.batch_size) % (len(train_labels) - self.conf.batch_size)
                indecies = range(train_offset[fold_index], train_offset[fold_index] + self.conf.batch_size)
                batch_data = [train_input[x] for x in indecies]
                


                #batch_data = train_input[train_offset[fold_index]:(train_offset[fold_index] + self.conf.batch_size)]
                #batch_labels = train_labels[offset:(offset + batch_size), :]
                train_offset[fold_index] = train_offset[fold_index] + self.conf.batch_size

            #feed_dict = {self.curr_step: step}

            #if step % self.conf.save_interval == 0:
            #    loss_value, images, labels, preds, summary, _ = self.sess.run(
            #        [self.reduced_loss,
            #         self.image_batch,
            #         self.label_batch,
            #         self.pred,
            #         self.total_summary,
            #         self.train_optimizer],
            #        feed_dict=feed_dict)
            #    self.summary_writer.add_summary(summary, step)
            #    self.save(self.saver, step)
            #else:
           # _, loss_value, _ = self.sess.run([self.train_optimizer, self.reduced_loss, self.train_prediction], feed_dict=feed_dict)

            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            #write_log('{:d}, {:.3f}'.format(step, loss_value), self.conf.logfile)


    def train_setup(self):
        tf.set_random_seed(self.conf.random_seed)

        
        # Load reader
        with tf.name_scope("create_inputs"):
            self.reader = SoundReaderKCrossValidation(
            self.conf.data_dir,
            self.conf.data_list,
            self.conf.input_size,
            self.conf.k_cross_val)
            self.net_input = tf.placeholder(tf.float32, shape=[self.conf.batch_size,1, self.conf.input_size,1])
            self.keep_prob = tf.placeholder(tf.float32)
            self.label_batch = tf.placeholder(tf.int32, shape=[self.conf.batch_size])


        #create network
        if (self.conf.encoder_name == 'DSRNet'):
            net = DSRNet(self.net_input, self.conf.num_classes, True,self.keep_prob)
            # Trainable Variables
            all_trainable = tf.trainable_variables()
            #todo - check if the regularization should be applied on all weights, uncluding fully connected

        else:
            print('encoder_name ERROR!')
            sys.exit(-1)

        # Network raw output
        logits = net.outputs  # [batch_size, #calses]

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(self.label_batch,self.conf.num_classes)))

        # L2 regularization
        l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]

        # Loss function
        self.reduced_loss = loss + tf.add_n(l2_losses)

        #Optimizer
        #learning rate configuration
        base_lr = tf.constant(self.conf.initial_learning_rate)
        self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.train.piecewise_constant(self.curr_step, self.conf.lr_schedule, [base_lr, 0.1*base_lr,0.01*base_lr])
        learning_rate = tf.cond(self.curr_step < tf.constant(self.conf.warmup), lambda: 0.1*base_lr, lambda: base_lr )
        #Nestrov optimizer with momentum
        optimizer = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum, use_nesterov= True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance of bach normailzaition
        with tf.control_dependencies(update_ops):
            self.train_optimizer = optimizer.minimize(self.reduced_loss)

        # Saver for storing checkpoints of the model
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

        # Loader for loading the pre-trained model
        self.loader = tf.train.Saver(var_list=tf.global_variables())

        #Predictions for the training, validation, and test data'''
        self.train_prediction = tf.nn.softmax(logits)


    def load(self, saver, filename):
        '''
        Load trained weights.
        ''' 
        saver.restore(self.sess, filename)
        print("Restored model parameters from {}".format(filename))

    def accuracy(self,predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])
    
    
    
    
    
    
    
    
    
    
    