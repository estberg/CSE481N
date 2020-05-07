from data_processing import MSASLDataLoader
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Softmax
import numpy as np

import i3d
import sonnet as snt
import json

# The path containing the information about samples. 
ANNOTATION_FILE_PATH = 'data/MS-ASL/frames/tiny.txt'
ANNOTATION_FILE_PATH_TRAIN = 'data/MS-ASL/frames/train.txt'

# The path containing the directories containing each samples frames. 
FRAMES_DIR_PATH = 'data/MS-ASL/frames/global_crops'

# The path containing checkponts
CHECKPOINT_PATHS = {
    'rgb_imagenet':'data/checkpoints/rgb_imagenet/model.ckpt',
    'rgb600':'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'rgb':'data/checkpoints/rgb_scratch/model.ckpt'
}

# KINETICS_LABEL_MAP_PATH = 'data/label_map.txt'

MSASL_LABEL_JSON = 'data/MS-ASL/meta/MSASL_classes100.json'

# We are taking the top-100 classes
NUM_CLASSES = 100 

# Number of epochs to train data
EPOCHS = 5

# Minimum frames seems to be 28. 
# The Train Generator will only take samples with at least this many frames. 
FRAME_LIMIT = 28

# Main method, this was super loose just to try to get the code to compile, probably not the best code to test the above model. 
def main():
    train_generator = MSASLDataLoader(ANNOTATION_FILE_PATH_TRAIN, FRAMES_DIR_PATH, 1, height=224, width=224, color_mode='rgb', shuffle=True, frames_threshold=FRAME_LIMIT)
    data_shape = train_generator.get_data_dim()

    # kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH)]
    msasl_classes = json.load(open(MSASL_LABEL_JSON))

    rgb_input = tf.compat.v1.placeholder(
        tf.float32,
        # i3d only accepts 224 x 224 image for now
        shape=(1, FRAME_LIMIT, 224, 224, 3))
    with tf.compat.v1.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    
    # The variable map is used to tell the saver which layers weights to restore. 
    # (the weights of the layers are all stored in tf variables)
    rgb_variable_map = {}
    for variable in tf.compat.v1.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')[len(''):]] = variable
            # rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d'):]] = variable    
    
    # We remove the logits layers from the variable map. We don't want to include these weights as we have a
    # different number of  classes.
    layers = rgb_variable_map.keys()
    layers_to_not_load = [layer for layer in layers if 'Logits' in layer]
    unloaded_layers_to_init = {}
    for layer in layers_to_not_load:
        unloaded_layers_to_init[layer] = rgb_variable_map.pop(layer)
    rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map, reshape=True)

    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)

    with tf.compat.v1.Session() as sess: 
        feed_dict = {}
        tf.compat.v1.global_variables_initializer()
        rgb_saver.restore(sess, CHECKPOINT_PATHS['rgb'])
        sess.run(tf.variables_initializer(list(unloaded_layers_to_init.values())))
        tf.compat.v1.logging.info('RGB checkpoint restored')
        rgb_variable_map.update(unloaded_layers_to_init)
        rgb_sample = train_generator[0][0]

        tf.compat.v1.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
        feed_dict[rgb_input] = rgb_sample

        out_logits, out_predictions = sess.run(
          [model_logits, model_predictions],
          feed_dict=feed_dict)

        out_logits = out_logits[0]
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]
        print('Norm of logits: %f' % np.linalg.norm(out_logits))
        print('\nTop classes and probabilities')
        for index in sorted_indices[:20]:
            print(out_predictions[index], out_logits[index], msasl_classes[index])
            
    # train the model
    # train_info = model.fit_generator(generator=train_generator, epochs=EPOCHS)
    
    # train_info.history includes loss and accuracy of the model from each epoch
    # print(train_info.history)

if __name__ == '__main__':
    main()
