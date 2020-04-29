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

# We are taking the top-100 classes
NUM_CLASSES = 100

# Number of epochs to train data
EPOCHS = 5

# Main method, this was super loose just to try to get the code to compile, probably not the best code to test the above model. 
def main():
    train_generator = MSASLDataLoader(ANNOTATION_FILE_PATH_TRAIN, FRAMES_DIR_PATH, 1, height=224, width=224, color_mode='rgb', shuffle=True)
    data_shape = train_generator.get_data_dim()
    print(train_generator.batch_size)
    rgb_input = tf.compat.v1.placeholder(
        tf.float32,
        #hard coded frame limit for 100, note that, i3d only accepts 224 x 224 image for now
        shape=(1, 100, 300, 256, 3))
    with tf.compat.v1.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.compat.v1.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
    rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map, reshape=True)
    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)

    # train the model
    # train_info = model.fit_generator(generator=train_generator, epochs=EPOCHS)
    
    # train_info.history includes loss and accuracy of the model from each epoch
    # print(train_info.history)

if __name__ == '__main__':
    main()
