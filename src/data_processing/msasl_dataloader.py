from tqdm import tqdm
import keras
import numpy as np
import sys
import random
import PIL

class MSASLDataLoader(keras.utils.Sequence):
    '''
    This is a MSASLDataLoader that loads the data as a keras sequence.

    TODO: This is probably not the best approach as each batch is loaded from images each time. 
    Ideally we would save the numpy representations of the images either the first time 
    they are loaded or in the preprocessing step to avoid this. 

    Essentially, it allows batch indexing on the data, and each epoch will randomy reorganize. 
    '''

    def __init__(self, file_annotations, frames_dir, batch_size, height, width, color_mode='rgb', shuffle=True, frames_threshold=0, num_classes=100):
        '''
        file_annotations : path
            List of files and their annotations in a text file
        frames_dir: path
            The path to the directory containing the frames directories
        batch_size: int
            The batch size
        width: int
            The width of the images
        height: int
            The height of the images
        color_mode: 'rgb', 'rgba', or 'grayscale'
            The color mode to read the images with
        shuffle: boolean
            Shuffle samples between epochs or not
        '''
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
        self.height = height
        self.width = width
        self.frames_threshold = frames_threshold
        self.num_classes = num_classes
        # Samples is a list of dictionaries containing the metadata for each sample
        # See the _make_samples() method for the keys in the dictionary
        self.samples, self.max_frames, self.min_frames = self._make_samples(file_annotations)
        self.frames_per_sample = max(self.frames_threshold, self.min_frames)
        trimmed_samples = []
        for sample in self.samples:
                if sample['duration'] >= self.frames_per_sample:
                    trimmed_samples.append(sample)
        self.samples = trimmed_samples
        self.indexes = np.arange(len(self.samples))
        self.on_epoch_end()

    def __len__(self):
        '''
        Returns the number of batches per epoch
        '''
        return int(np.floor(len(self.samples) / self.batch_size))
    
    def __getitem__(self, index):
        '''
        Reads in and returns one batch of data.

        Note that this can be used as:
            data_generator = MSASLDataLoader(...)
            data_generator[0] # will return the first batch of data

        Returns:
            X : tensor (batch_size, max_frames, height, width, color_channels)
                the data of the batch
                TODO: Note that this is max_frames length, so there will be padded frames
                of all zeros in many cases. 
            y : tensor (batch_size, )
                the labels of the batch
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_samples_idxs = [self.samples[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(batch_samples_idxs)

        return X, y
    
    def _data_generation(self, batch_samples_idxs):
        """
        Generates data containing batch_size samples.

        Returns:
            X : tensor (batch_size, max_frames, height, width, color_channels)
                the data of the batch
                TODO: Note that this is max_frames length, so there will be padded frames
                of all zeros in many cases.
            y : tensor (batch_size, )
                the labels of the batch
        """
        # X : (batch_size, max_frames, height, width, color_channels)
        # paddings : (batch_size, )
        # y : (batch_size, )
        # Initialization
        X = np.empty((self.batch_size, *self.get_data_dim()))
        paddings = np.empty((self.batch_size), dtype=int)
        y = np.empty((self.batch_size, self.num_classes), dtype=int)

        # Load Images, (Do Preprocessing?)
        for idx, sample in tqdm(enumerate(batch_samples_idxs), desc='Loading Batch Images'):
            # TODO: add a step that saves the loaded numpy array 
            # so the image doesn't need to be loaded and padded 
            # each epoch. Then also add a function in main (or whereever)
            # to delete these files after a run (preprocessing, etc, could change)
            start = random.randrange(sample['start'], sample['end'] - (self.frames_per_sample - 1))
            end = start + self.frames_per_sample
            for frame_idx in range(start, end):
                num = f'{(frame_idx + 1):05}'
                path = self.frames_dir + '/' + sample['rel_images_dir'] + '/img_' + num + '.jpg'
                img = keras.preprocessing.image.load_img(
                    path, grayscale=False, color_mode=self.color_mode, target_size=(self.height, self.width),
                    interpolation='nearest'
                )
                x_ = keras.preprocessing.image.img_to_array(img, data_format='channels_last', dtype=int)
                X[idx, frame_idx - start, ] = x_

            # Store class
            y[idx][sample['label']] = 1
            
            # cannot return now, but might come in handy later (?)
            paddings[idx] = self.max_frames - frame_idx

        return X, y
    
    def _make_samples(self, file_annotations):
        """
        file_annotations: path
            Path to the txt file containing the metadata for the samples
        
        Returns:
            sample: list of dictionaries
                Contains the parsed metadata for the samples
            max_frames:
                The maximum number of frames in a sample
        """
        max_frames = 0
        min_frames = None
        samples = []
        with open(file_annotations) as f:
            samples_list = f.readlines()
            for sample in samples_list:
                annotations = sample.split()
                labeled_annotations = dict()
                labeled_annotations['rel_images_dir'] = annotations[0]
                labeled_annotations['label'] = int(annotations[1])
                # TODO: I am not sure about this
                # start = int(annotations[2])
                # end = int(annotations[3])
                start = int(annotations[4])
                end = int(annotations[5])
                duration = end - start
                labeled_annotations['start'] = start
                labeled_annotations['end'] = end
                labeled_annotations['duration'] = duration
                # print(duration)
                labeled_annotations['total_duration'] = int(annotations[5])
                labeled_annotations['fps'] = float(annotations[6])
                samples.append(labeled_annotations)
                if duration > max_frames:
                    max_frames = duration
                if min_frames is None or duration < min_frames:
                    min_frames = duration
        return samples, max_frames, min_frames
    
    def get_data_dim(self):
        """
        Returns dimension of one data sample (this excludes batch size but 
        is the shape of one element of a batch)
        """
        return (self.frames_per_sample, self.height, self.width, self.color_channels)
    
    def on_epoch_end(self):
        """
        If set at initialization randomly sort samples after each epoch
        """
        if self.shuffle == True:
            random.shuffle(self.samples)
