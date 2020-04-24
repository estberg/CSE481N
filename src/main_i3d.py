from data_processing import MSASLDataLoader
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Softmax
import numpy as np

import i3d

# The path containing the information about samples. 
ANNOTATION_FILE_PATH = 'data/MS-ASL/frames/tiny.txt'
ANNOTATION_FILE_PATH_TRAIN = 'data/MS-ASL/frames/train.txt'

# The path containing the directories containing each samples frames. 
FRAMES_DIR_PATH = 'data/MS-ASL/frames/global_crops'

# Because I think we are taking the top-100 classes
NUM_CLASSES = 100

# Number of epochs to train data
EPOCHS = 5

# Super Naive Approach
# This model has no weights, so it actually does not neet to be compiled or trained.
# It just takes the maxes and sums of various values to essentially random choose a label. 

# Note: Unfortunately, I didn't have time to get it running, but I think you could use predict() 
# on it in main to test it, passing one sample. You could get these samples by using a MSASLDataLoader with
# with a big batch size, and then index to get each sample and pass it into predict. 
def get_model(train_generator, data_shape):
    model = i3d.InceptionI3d(NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    return model

# Main method, this was super loose just to try to get the code to compile, probably not the best code to test the above model. 
def main():
    train_generator = MSASLDataLoader(ANNOTATION_FILE_PATH_TRAIN, FRAMES_DIR_PATH, 1, height=300, width=256, color_mode='rgb', shuffle=True)
    data_shape = train_generator.get_data_dim()
    print(train_generator.batch_size)
    model = get_model(train_generator, data_shape)
    rgb_input = tf.placeholder(
        tf.float32,
#hard coded frame limit for 100
        shape=(1, 100, 300, 256, 3))
    rgb_logits, _ = model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)

    # train the model
    train_info = model.fit_generator(generator=train_generator, epochs=EPOCHS)
    
    # train_info.history includes loss and accuracy of the model from each epoch
    print(train_info.history)

if __name__ == '__main__':
    main()
