# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:26:01 2018

@author: Avi
"""
import numpy as np
import tensorflow as tf
import csv
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

def read_sounds_from_disk(input_queue, input_size): # optional pre-processing arguments
    """Read one sound and its corresponding label and pre-process it.
    
    Args:
      input_queue: tf queue with paths to the sounds and its labels.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """
    #sound = tf.read_file(input_queue[0])
    audio_binary = tf.read_file(input_queue[0])
    desired_channels = 1
    wav_decoder = contrib_audio.decode_wav(audio_binary,desired_channels=desired_channels)
    sound  = wav_decoder.audio.flatten()
    
    label = input_queue[1]
    
    ##TODO implement preprocessing
    
#    img = tf.image.decode_jpeg(img_contents, channels=3)
#    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
#    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
#    # Extract mean.
#    img -= img_mean
#
#    label = tf.image.decode_png(label_contents, channels=1)
#
#    if input_size is not None:
#        h, w = input_size
#
#        # Randomly scale the images and labels.
#        if random_scale:
#            img, label = image_scaling(img, label)
#
#        # Randomly mirror the images and labels.
#        if random_mirror:
#            img, label = image_mirroring(img, label)
#
#        # Randomly crops the images and labels.
#        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return sound, label


class SoundReader(object):
    '''Generic SoundReader which reads sound files and their labels
       from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: all files will be resized to that value.
          coord: TensorFlow queue coordinator.
        '''
        
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        
        self.sounds, self.labels, self.is_ec10 = read_labeled_sound_list(self.data_dir, self.data_list)

        self.sounds = tf.convert_to_tensor(self.sounds, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        self.queue = tf.train.slice_input_producer([self.sounds, self.labels],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.sound, self.label = read_sounds_from_disk(self.queue, self.input_size) 

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        sound_batch, label_batch = tf.train.batch([self.sound, self.label],
                                                  num_elements)
        return sound_batch, label_batch