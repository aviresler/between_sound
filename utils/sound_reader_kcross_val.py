import numpy as np
import csv
import scipy.io.wavfile


def read_from_disk_and_preprocess(data_dir, data_list,zero_padding):
    """Reads txt file containing paths to sound files and ground truth labels.
    Loads all the data into memory, after basic preprocessing

    Args:
      zero_padding: amount of zeros to put in the beginning/end of each audio signal.
      data_dir: path to the directory of all the audio files
      data_list: path to the csv file with the data description

    Returns:
      Lists with all file names, labels, and is_esc10 (esc10 is a smaller dataset).
    """
    with open(data_list) as csvfile:
        reader = csv.DictReader(csvfile)
        labels = []
        is_esc10 = []
        data_list = []
        for row in reader:
            labels.append(int(row['target']))
            is_esc10.append(row['esc10'])
            [_, data] = scipy.io.wavfile.read(data_dir + row['filename'])
            # conversion to float
            data = data.astype(float)
            # normalization between -1 to 1
            data = data / (1 << 15)
            # remove 0 from start/end
            data = np.trim_zeros(data)
            # pad with T/2 zeros from start&end
            data = np.pad(data, (zero_padding, zero_padding), 'constant', constant_values=(0, 0))
            data_list.append(data)

    return data_list, labels, is_esc10

def convert_to_array_and_crop_randomly(data_list, random_index):
    """Reads txt file containing paths to sound files and ground truth labels.
    Loads all the data into memory, after basic preprocessing

    Args:
      zero_padding: amount of zeros to put in the beginning/end of each audio signal.
      data_dir: path to the directory of all the audio files
      data_list: path to the csv file with the data description

    Returns:
      Lists with all file names, labels, and is_esc10 (esc10 is a smaller dataset).
    """
    with open(data_list) as csvfile:
        reader = csv.DictReader(csvfile)
        labels = []
        is_esc10 = []
        data_list = []
        for row in reader:
            labels.append(int(row['target']))
            is_esc10.append(row['esc10'])
            [_, data] = scipy.io.wavfile.read(data_dir + row['filename'])
            # conversion to float
            data = data.astype(float)
            # normalization between -1 to 1
            data = data / (1 << 15)
            # remove 0 from start/end
            data = np.trim_zeros(data)
            # pad with T/2 zeros from start&end
            data = np.pad(data, (zero_padding, zero_padding), 'constant', constant_values=(0, 0))
            data_list.append(data)

    return data_list, labels, is_esc10





class SoundReaderKCrossValidation(object):
    '''Generic SoundReader which reads sound files and their labels.
    It splits the data into k cross validation section (after the needed preprocessing
    '''

    def __init__(self, data_dir, data_list, input_size, k):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: all files will be resized to that value.
          k: number of cross validation section (typically 5)
        '''
        
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.k = k
        self.zero_padding = int(self.input_size/2+1)
        self.data, self.labels, self.is_ec10 = read_from_disk_and_preprocess(self.data_dir,self.data_list,self.zero_padding )
        self.fold_length = int(len(self.data) / self.k)




    def split_according_to_fold(self, fold_index):
        """splits data into training/validation according to k-cross validation method

        Args:
          data: list of all the data
          fold_index: this fold will be used for validation, and the rest for training

        Returns:
         train_list and validation list according to the required validation fold
        """

        index = np.arange(len(self.data))
        lower_bound = index >= fold_index * self.fold_length
        upper_bound = index < (fold_index + 1) * self.fold_length
        cv_region = lower_bound * upper_bound
        cv_indecies = index[np.nonzero(cv_region)]
        train_indecies = index[np.nonzero(~cv_region)]
        #print(cv_region)

        cv_data = [self.data[x] for x in cv_indecies]
        train_data = [self.data[x] for x in train_indecies]

        cv_labels = [self.labels[x] for x in cv_indecies]
        train_labels = [self.labels[x] for x in train_indecies]

        return (train_data, train_labels), (cv_data, cv_labels)
