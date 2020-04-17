from tqdm import tqdm
import keras
import numpy as np
import sys
import random
import PIL

class MSASLDataLoader(keras.utils.Sequence):

    '''
    file_annotations : path
        List of files and their annotations in a text file
    '''
    def __init__(self, file_annotations, frames_dir, batch_size, width, height, color_mode='rgb', shuffle=True):
        self.file_annotations = file_annotations
        self.frames_dir = frames_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.color_mode = color_mode
        if color_mode == 'rgb':
            self.color_channels = 3
        elif color_mode == 'rgba':
            self.color_channels = 4
        elif color_mode == 'grayscale':
            self.color_channels = 1
        else:
            raise Exception('Invalid Color Mode')
        self.width = width
        self.height = height
        self.samples, self.max_frames = self._make_samples(file_annotations)
        self.indexes = np.arange(len(self.samples))
        self.on_epoch_end()

    def __len__(self):
        'Returns the number of batches per epoch'
        return int(np.floor(len(self.samples) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_samples_idxs = [self.samples[k] for k in indexes]

        # Generate data
        X, y, paddings = self.__data_generation(batch_samples_idxs)

        return X, y, paddings
    
    def __data_generation(self, batch_samples_idxs):
        'Generates data containing batch_size samples' 
        # X : (batch_size, max_frames, color_channels, width, height)
        # paddings : (batch_size, )
        # y : (batch_size, )
        # Initialization
        X = np.empty((self.batch_size, *self.get_data_dim()))
        paddings = np.empty((self.batch_size), dtype=int)
        y = np.empty((self.batch_size), dtype=int)

        # Load Images, (Do Preprocessing?)
        for idx, sample in tqdm(enumerate(batch_samples_idxs), desc='Loading Batch Images'):
            # TODO: add a step that saves the loaded numpy array 
            # so the image doesn't need to be loaded and padded 
            # each epoch. Then also add a function in main (or whereever)
            # to delete these files after a run (preprocessing, etc, could change)
            start = sample['start']
            end = sample['end']
            for frame_idx in range(start, end):
                num = f'{frame_idx:05}'
                path = self.frames_dir + '/' + sample['rel_images_dir'] + '/img_' + num + '.jpg'
                img = keras.preprocessing.image.load_img(
                    path, grayscale=False, color_mode=self.color_mode, target_size=(self.height, self.width),
                    interpolation='nearest'
                )
                x_ = keras.preprocessing.image.img_to_array(img, data_format='channels_first', dtype=int)
                print('frame_idx', frame_idx)
                print('start', start)
                print('X.shape', X.shape)
                X[idx, frame_idx - start, ] = x_

            # Store class
            y[idx] = sample['label']
            
            # cannot return now, but might come in handy later (?)
            paddings[idx] = self.max_frames - frame_idx

        return X, y
    
    def _make_samples(self, file_annotations):
        max_frames = 0
        samples = []
        with open(file_annotations) as f:
            samples_list = f.readlines()
            for sample in samples_list:
                annotations = sample.split()
                labeled_annotations = dict()
                labeled_annotations['rel_images_dir'] = annotations[0]
                labeled_annotations['label'] = int(annotations[1])
                start = int(annotations[2])
                end = int(annotations[3])
                duration = end - start
                labeled_annotations['start'] = start
                labeled_annotations['end'] = end
                labeled_annotations['duration'] = duration
                labeled_annotations['total_duration'] = int(annotations[5])
                labeled_annotations['fps'] = float(annotations[6])
                samples.append(labeled_annotations)
                if duration > max_frames:
                    max_frames = duration
        return samples, max_frames
    
    def get_data_dim(self):
        'Returns dimension of one data sample'
        return (self.max_frames, self.color_channels, self.width, self.height)
    
    def on_epoch_end(self):
        'Randomly sort samples after each epoch'
        if self.shuffle == True:
            random.shuffle(self.samples)