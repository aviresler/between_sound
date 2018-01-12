# -*- coding: utf-8 -*-

import tensorflow as tf
from network_class import *
from utils.sound_reader import SoundReader

class Model(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        
	# train
    def train(self):
        self.train_setup()

        self.sess.run(tf.global_variables_initializer())

        #TODO Load the pre-trained model if provided
        #if self.conf.pretrain_file is not None:
        #    self.load(self.loader, self.conf.pretrain_file)
            
        #Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)


#		# Train!
#		for step in range(self.conf.num_steps+1):
#			start_time = time.time()
#			feed_dict = { self.curr_step : step }
#
#			if step % self.conf.save_interval == 0:
#				loss_value, images, labels, preds, summary, _ = self.sess.run(
#					[self.reduced_loss,
#					self.image_batch,
#					self.label_batch,
#					self.pred,
#					self.total_summary,
#					self.train_op],
#					feed_dict=feed_dict)
#				self.summary_writer.add_summary(summary, step)
#				self.save(self.saver, step)
#			else:
#				loss_value, _ = self.sess.run([self.reduced_loss, self.train_op],
#					feed_dict=feed_dict)
#
#			duration = time.time() - start_time
#			print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
#			write_log('{:d}, {:.3f}'.format(step, loss_value), self.conf.logfile)
#
		# finish
        self.coord.request_stop()
        self.coord.join(threads)

    def train_setup(self):
        tf.set_random_seed(self.conf.random_seed)
        
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()
        
        # Load reader
        with tf.name_scope("create_inputs"):
            reader = SoundReader(
            self.conf.data_dir,
            self.conf.data_list,
            self.conf.input_size,
            self.coord,
            self.sess)
            self.sound_batch, self.label_batch = reader.dequeue(self.conf.batch_size)
        
        print(self.sound_batch.shape)
        #create network
        if (self.conf.encoder_name == 'DSRNet'):
            net = DSRNet(self.sound_batch, self.conf.num_classes, True)
            
            

    
    # Create network
#    net = DSRNet(self.image_batch, self.conf.num_classes, True)

    def load(self, saver, filename):
        '''
        Load trained weights.
        ''' 
        saver.restore(self.sess, filename)
        print("Restored model parameters from {}".format(filename))
    
    
    
    
    
    
    
    
    
    
    