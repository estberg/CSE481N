import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os

# suppress tf INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from data_processing import MSASLDataLoader
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Softmax
import numpy as np

import i3d
import sonnet as snt
import json
from tqdm import tqdm

import code

# The path containing the information about samples. 
ANNOTATION_FILE_PATH_TINY = 'data/MS-ASL/frames224/tiny100.txt'
ANNOTATION_FILE_PATH_TRAIN = 'data/MS-ASL/frames224/train100.txt'
ANNOTATION_FILE_PATH_VAL = 'data/MS-ASL/frames224/val100.txt'

# The path containing the directories containing each samples frames. 
FRAMES_DIR_PATH = 'data/MS-ASL/frames224/global_crops'

# The path containing checkponts
CHECKPOINT_PATHS = {
    'rgb_imagenet':'data/checkpoints/rgb_imagenet/model.ckpt',
    'rgb600':'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'rgb':'data/checkpoints/rgb_scratch/model.ckpt'
}

NAME = 'modelDecayMMTacc0.20041-61'
WEIGHTS_PATH = 'data/checkpoints/ms_asl/' + NAME

# KINETICS_LABEL_MAP_PATH = 'data/label_map.txt'

MSASL_LABEL_JSON = 'data/MS-ASL/meta/MSASL_classes100.json'

# We are taking the top-100 classes
NUM_CLASSES = 100 

# Process one sample at a time
BATCH_SIZE = 8

# Random Image Flipping
FLIPPING = True

# Droupout keep rate (1 - dropout_rate)
DROPOUT_KEEP_PROB = 0.5

# Learning rate for the optimizer
LEARNING_RATE = 0.01

# Momentum for the optimizer
MOMENTUM = 0.9

# Parameters for Adam optimizer
ADAM_INIT_LR = 0.01
ADAM_EPS = 1e-3
ADAM_WEIGHT_DECAY = 1e-7

# Number of epochs to train data
EPOCHS = 100

# Minimum frames seems to be 28. 
# The Train Generator will only take samples with at least this many frames. 
FRAME_LIMIT = 64

# Main method makes a train and validation set generator
def main():
    train_generator = MSASLDataLoader(ANNOTATION_FILE_PATH_TRAIN, FRAMES_DIR_PATH, batch_size=BATCH_SIZE, height=224, width=224, color_mode='rgb', shuffle=True, frames_threshold=FRAME_LIMIT, num_classes=NUM_CLASSES, flipping=FLIPPING)
    validation_generator = MSASLDataLoader(ANNOTATION_FILE_PATH_VAL, FRAMES_DIR_PATH, batch_size=BATCH_SIZE, height=224, width=224, color_mode='rgb', shuffle=True, frames_threshold=FRAME_LIMIT, num_classes=NUM_CLASSES, flipping=FLIPPING)
    data_shape = train_generator.get_data_dim()

    print("Training Samples", len(train_generator ) * BATCH_SIZE)

    # kinetics_classes = [x.strip() for x in open(LABEL_MAP_PATH)]
    msasl_classes = json.load(open(MSASL_LABEL_JSON))

    rgb_input = tf.compat.v1.placeholder(
        tf.float32,
        # i3d only accepts 224 x 224 image for now
        shape=(BATCH_SIZE, FRAME_LIMIT, 224, 224, 3))

    load_model_and_test(train_generator, validation_generator, msasl_classes, rgb_input)

def load_model_and_test(train_generator, validation_generator, msasl_classes, rgb_input):
    '''
    Trains for EPOCH on the train_generator's data and tests the validation set. 
    '''
    with tf.compat.v1.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=DROPOUT_KEEP_PROB)
    
    # The variable map is used to tell the saver which layers weights to restore. 
    # (the weights of the layers are all stored in tf variables)
    rgb_variable_map = {}
    for variable in tf.compat.v1.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')[len(''):]] = variable
            # rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d'):]] = variable    
    
    # We remove the logits layers from the variable map. We don't want to include these weights as we have a
    # different number of  classes.
    # layers = rgb_variable_map.keys()
    # layers_to_not_load = [layer for layer in layers if 'Logits' in layer]
    # unloaded_layers_to_init = {}
    # for layer in layers_to_not_load:
    #     unloaded_layers_to_init[layer] = rgb_variable_map.pop(layer)
    rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map, reshape=True)

    with tf.compat.v1.Session() as sess: 
        rgb_saver.restore(sess, WEIGHTS_PATH)
        tf.compat.v1.logging.info('RGB checkpoint restored')
            
        validate(sess, train_generator, rgb_model, rgb_input, 'Train', msasl_classes)
        validate(sess, validation_generator, rgb_model, rgb_input, 'Validation', msasl_classes)

def validate(sess, validation_generator, rgb_model, rgb_input, data_set, msasl_classes):
    # building phase (tensorflow placeholders) for model prediction
    val_logits, _ = rgb_model(
      rgb_input, is_training=False, dropout_keep_prob=DROPOUT_KEEP_PROB)
    # TODO: val_labels is not needed now, may need to delete later
    val_labels = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
    val_predictions = tf.nn.softmax(val_logits)

    # placeholders for predicted and expected labels
    out_labels = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
    true_labels_one_hot = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])

    # placeholders for accuracy calculation
    true_labels = tf.argmax(true_labels_one_hot, axis=1)
    predictions = tf.argmax(val_predictions, axis=1)
    correct_prediction = tf.equal(predictions, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    top_5_accuracy = tf.reduce_mean(tf.cast(tf.math.in_top_k(targets=true_labels, predictions=val_predictions, k=5), tf.float32))
    top_10_accuracy = tf.reduce_mean(tf.cast(tf.math.in_top_k(targets=true_labels, predictions=val_predictions, k=10), tf.float32))

    # processes one sample
    def step(samples, labels):
        feed_dict = {}
        feed_dict[rgb_input] = samples
        feed_dict[true_labels_one_hot] = labels
        out_logits, step_accuracy, step_top_5, step_top_10, step_true_labels, step_predictions = sess.run(
            [val_logits, accuracy, top_5_accuracy, top_10_accuracy, true_labels, predictions],
            feed_dict=feed_dict)
        step_confusion_matrix = tf.math.confusion_matrix(labels=step_true_labels, predictions=step_predictions, num_classes=100)
        return step_accuracy, step_top_5, step_top_10, step_confusion_matrix

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    batches = 0 
    avg_accuracy = 0
    avg_top_5 = 0
    avg_top_10 = 0
    for images, labels in tqdm(validation_generator, desc=data_set):
        batch_accuracy, batch_top_5, batch_top_10, batch_confusion_matrix = step(images, labels)
        avg_accuracy += batch_accuracy
        avg_top_5 += batch_top_5
        avg_top_10 += batch_top_10
        batches += 1
        confusion_matrix += batch_confusion_matrix
    
    avg_accuracy = avg_accuracy / batches
    avg_top_5 = avg_top_5 / batches
    avg_top_10 = avg_top_10 / batches
    print(data_set + " Accruacy:" + str(avg_accuracy) + " Top 5:" + str(avg_top_5) + " Top 10:" + str(avg_top_10))
    
    fig, ax = plt.subplots(figsize=(30,30))
    df_cm = pd.DataFrame(confusion_matrix.eval(), index = [i for i in msasl_classes], columns = [i for i in msasl_classes])
    sn.heatmap(df_cm, annot=True, ax=ax)
    fig.savefig(os.getcwd() + "/graphs/" + NAME + data_set + 'confusion.jpg')
    return avg_accuracy

if __name__ == '__main__':
    main()
