import argparse
import os
import tensorflow as tf
from model_sound import Model



"""
This script defines hyperparameters.
"""



def configure():
	flags = tf.app.flags

	# training
	flags.DEFINE_integer('num_steps', 20000, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', 1000, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'DSRNet', 'name of pre-trained model')
	flags.DEFINE_string('pretrain_file', '', 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', 'dataset\ESC-50-master\meta\esc50.csv', 'training data list filename')

	# validation
	flags.DEFINE_integer('valid_step', 20000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 1449, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', './dataset/val.txt', 'validation data list filename')

	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', 'output', 'directory for saving outputs')
	flags.DEFINE_integer('test_step', 20000, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', 1449, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list', './dataset/val.txt', 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

	# data
	flags.DEFINE_string('data_dir', 'dataset/ESC-50-master/audio','path to data directory')
	flags.DEFINE_integer('batch_size', 64, 'training batch size')
	flags.DEFINE_integer('input_size', 66650, 'input image width')
	flags.DEFINE_integer('num_classes', 50, 'number of classes')
	
	# log
	flags.DEFINE_string('modeldir', 'model', 'model directory')
	flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log', 'training log directory')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='train',
		help='actions: train, test, or predict')
	args = parser.parse_args()

	if args.option not in ['train', 'test', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")
	else:
		# Set up tf session and initialize variables. 
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)
		sess = tf.Session()
		# Run
		model = Model(sess, configure())
		getattr(model, args.option)()


if __name__ == '__main__':
	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = '7'
	tf.app.run()
