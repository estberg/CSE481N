from data_processing import MSASLDataLoader
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Softmax
import numpy as np

# The path containing the information about samples. 
ANNOTATION_FILE_PATH = 'data/MS-ASL/frames/tiny.txt'

# The path containing the directories containing each samples frames. 
FRAMES_DIR_PATH = 'data/MS-ASL/frames/global_crops'

# Super Naive Approach
# This model has no weights, so it actually does not neet to be compiled or trained.
# It just takes the maxes and sums of various values to essentially random choose a label. 

# Note: Unfortunately, I didn't have time to get it running, but I think you could use predict() 
# on it in main to test it, passing one sample. You could get these samples by using a MSASLDataLoader with
# with a big batch size, and then index to get each sample and pass it into predict. 
def get_model(data_shape):

    input_layer = Input(shape=data_shape, name="input_layer")
    # data_shape = (max_frames, color_channels, width, height)
    layer_1 = Lambda(lambda x: tf.math.reduce_max(x, axis=0), input_shape=data_shape, output_shape=data_shape[1:], name="layer_1")(input_layer)
    # data_shape = (color_channels, width, height)
    layer_2 = Lambda(lambda x: tf.math.reduce_max(x, axis=0), input_shape=data_shape[1:], output_shape=data_shape[2:], name="layer_2")(layer_1)
    # data_shape = (width, height)
    layer_3 = Lambda(lambda x: tf.math.reduce_sum(x, axis=0), input_shape=data_shape[2:], output_shape=data_shape[3:], name="layer_3")(layer_2)
    # data_shape = (height)
    layer_4 = Lambda(lambda x: tf.math.reduce_sum(x, axis=0) % 1000, input_shape=data_shape[3:], output_shape=[1,], name="layer_4")(layer_3)

    model = Model(input_layer, layer_4, name="model")
    model.summary()
    return model


# Main method, this was super loose just to try to get the code to compile, probably not the best code to test the above model. 
def main():
    train_generator = MSASLDataLoader(ANNOTATION_FILE_PATH, FRAMES_DIR_PATH, 1, 300, 256, color_mode='rgb', shuffle=True)
    data_shape = train_generator.get_data_dim()
    model = get_model(data_shape)

if __name__ == '__main__':
    main()