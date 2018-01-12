# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:26:01 2018

@author: Avi
"""
import numpy as np
import tensorflow as tf
import csv
import scipy.io.wavfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio


def read_labeled_sound_list(data_dir, data_list):
    """Reads txt file containing paths to sound files and ground truth labels.
    
    Args:
      data_dir: path to the directory with sound files
      data_list: path to the csv file with database description.
       
    Returns:
      Lists with all file names, labels, and is_esc10 (esc10 is a smaller dataset).
    """
    with open(data_list) as csvfile:
        reader = csv.DictReader(csvfile)
        sound_files = []
        labels = []
        is_esc10 = []
        for row in reader:
            sound_files.append(data_dir + row['filename'])
            labels.append(int(row['target']))
            is_esc10.append(row['esc10'])
    
    return sound_files, labels, is_esc10

def read_sounds_from_disk(input_queue, input_size,sess): # optional pre-processing arguments
    """Read one sound and its corresponding label and pre-process it.

    Args:
      input_queue: tf queue with paths to the sounds and its labels.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.

    Returns:
      Two tensors: the decoded sound and its label.
    """

    audio_binary = tf.read_file(input_queue[0])
    #print(audio_binary)
    desired_channels = 1
    wav_decoder = contrib_audio.decode_wav(audio_binary, desired_channels=desired_channels)
    sound = wav_decoder.audio
    sound.set_shape([ input_size, 1])
    sound = tf.transpose(sound)
    sound = tf.expand_dims(sound, -1)

    label = input_queue[1]

    #TODO implement preprocessing

    return sound, label


class SoundReader(object):
    '''Generic SoundReader which reads sound files and their labels
       from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, coord,sess):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: all files will be resized to that value.
          sess: tensorflow session
          coord: TensorFlow queue coordinator.
        '''
        
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        self.sess = sess
        
        self.sounds, self.labels, self.is_ec10 = read_labeled_sound_list(self.data_dir, self.data_list)

        self.sounds = tf.convert_to_tensor(self.sounds, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        self.queue = tf.train.slice_input_producer([self.sounds, self.labels],
                                                   shuffle=False) # not shuffling if it is val
        self.sound, self.label = read_sounds_from_disk(self.queue, self.input_size,self.sess)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        sound_batch, label_batch = tf.train.batch([self.sound, self.label],
                                                  num_elements)
        return sound_batch, label_batch