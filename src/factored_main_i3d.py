from data_processing import MSASLDataLoader
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Softmax
import numpy as np

import i3d
import sonnet as snt
import json
from tqdm import tqdm

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

NEW_CHECKPOINT_PATHS = 'data/checkpoints/ms_asl/model'
NAME = 'DecayMMTShortSamplesFlipImages'

# KINETICS_LABEL_MAP_PATH = 'data/label_map.txt'

MSASL_LABEL_JSON = 'data/MS-ASL/meta/MSASL_classes100.json'

# We are taking the top-100 classes
NUM_CLASSES = 100 

BATCH_SIZE = 8

# Random Image Flipping
FLIPPING = True

# Droupout keep rate (1 - dropout_rate)
DROPOUT_KEEP_PROB = 0.7

# Learning rate for the optimizer
LEARNING_RATE = 0.001

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

    train_from_kinetics_weights(train_generator, validation_generator, msasl_classes, rgb_input)


def train_from_kinetics_weights(train_generator, validation_generator, msasl_classes, rgb_input):
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
        
        # Restore all the layers but the logits from the shared weights.
        rgb_saver.restore(sess, CHECKPOINT_PATHS['rgb'])
        tf.compat.v1.logging.info('RGB checkpoint restored')

        # Initialize the logits (final layer). Not sure exactly how they will be initialized.
        # TODO: Look into how the logits will be initialized. Could try different approaches.
        sess.run(tf.compat.v1.variables_initializer(list(unloaded_layers_to_init.values())))
        
        # Preparing a new saver on all the layers to save weights as we train.
        # TODO: Use this saver to save in training.
        rgb_variable_map.update(unloaded_layers_to_init)
        rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map, reshape=True)

        # input and output shapes
        rgb_logits, _ = rgb_model(
          rgb_input, is_training=True, dropout_keep_prob=DROPOUT_KEEP_PROB)
        rgb_labels = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
        
        
        # TODO: Try with a more reasonable learning rate
        # global_step and decayed_lr can be used to decay the learning rate exponentially
        global_step = tf.compat.v1.placeholder(tf.int32)
        decayed_lr = tf.compat.v1.train.exponential_decay(learning_rate=ADAM_INIT_LR, global_step=global_step, decay_steps=5, decay_rate=0.95)

        # Loss and optimizer to use
        # TODO: Try with different loss functions and optimizers
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=rgb_logits, labels=rgb_labels)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=rgb_logits, labels=rgb_labels))
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=MOMENTUM)
        # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decayed_lr, epsilon=ADAM_EPS)
        minimize = optimizer.minimize(loss)
        sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))

        # One step or batch of training.
        def step(samples, labels, i):
            """Performs one optimizer step on a single mini-batch."""
            feed_dict[rgb_input] = samples
            feed_dict[rgb_labels] = labels
            feed_dict[global_step] = i
            result = sess.run(
                [loss, minimize],
                feed_dict=feed_dict)
            return result

        # TODO: Should this be running loss and need to be fixed?
        # One epoch of training
        def epoch(data_generator, i):
            for images, labels in tqdm(data_generator, desc='EPOCH' + str(i)):
                result = step(images, labels, i)
            data_generator.on_epoch_end()
            print("Loss" + str(result[0]))
            return result[0]
        
        # val_accuracy_prior = 0

        # summary_saver keeps track of epoch_num, loss, traning acc., validation acc.
        summary_saver = []
        for i in range(EPOCHS):
            epoch_loss = epoch(train_generator, i)
            summary_saver.append([i + 1, epoch_loss])
            if i % 10 == 0:
                train_accuracy = validate(sess, train_generator, rgb_model, rgb_input, 'Train')
                summary_saver[i].append(train_accuracy)
            # evaluate validation set for every epoch
            val_accuracy = validate(sess, validation_generator, rgb_model, rgb_input, 'Validation')
            summary_saver[i].append(val_accuracy)
            rgb_saver.save(sess, NEW_CHECKPOINT_PATHS + NAME + 'acc' + ('%.5f' % val_accuracy), global_step=i)
            if i % 5 == 0:
                print_train_summary(summary_saver)
            # # if validation accuracy starts decreasing, decrease the learning rate
            # if val_accuracy_prior < val_accuracy:
            #     decayed_lr = decayed_lr / 10
            #     optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=MOMENTUM)
            #     # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decayed_lr, epsilon=ADAM_EPS)
            #     minimize = optimizer.minimize(loss)
            #     sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))

def print_train_summary(summary_saver):
    # print summary which helps us create table
    for result in summary_saver:
        for i in range(len(result)):
            if i != 0:
                print(', ', end='')
            if len(result) == 3 and i == 2:
                print(', ', end='')
            print(str(result[i]), end='')
        print()

def validate(sess, validation_generator, rgb_model, rgb_input, data_set):
    # building phase (tensorflow placeholders) for model prediction
    val_logits, _ = rgb_model(
      rgb_input, is_training=False, dropout_keep_prob=DROPOUT_KEEP_PROB)
    # TODO: val_labels is not needed now, may need to delete later
    val_labels = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
    val_predictions = tf.nn.softmax(val_logits)

    # placeholders for predicted and expected labels
    out_labels = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])
    true_labels = tf.compat.v1.placeholder(tf.float32, [None, NUM_CLASSES])

    # placeholders for accuracy calculation
    correct_prediction = tf.equal(tf.argmax(val_predictions, axis=1), tf.argmax(true_labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TODO: Fix bug in accuracy calculation
    def step(samples, labels):
        feed_dict = {}
        feed_dict[rgb_input] = samples
        feed_dict[true_labels] = labels
        out_logits, step_accuracy = sess.run(
            [val_logits, accuracy],
            feed_dict=feed_dict)
        return step_accuracy
        # out_logits, out_predictions = sess.run(
        #     [val_logits, val_predictions],
        #     feed_dict=feed_dict)
        # sorted_indices = np.argsort(out_predictions, axis=1)[::-1]
        # print(sorted_indices.shape)
        # print("sorted indices:")
        # print(sorted_indices)
        # print("logits: {}".format(out_logits))
        # print("out pred: {}".format(out_predictions))
        # print("labels: {}".format(labels))
        # print(out_logits.shape)
        # print(out_predictions.shape)
        # print(labels.shape)
        # batch_accuracy, batch_accuracy_opp = tf.compat.v1.metrics.accuracy(out_labels, true_labels)
        # sess.run(tf.compat.v1.local_variables_initializer())
        # v = sess.run([batch_accuracy, batch_accuracy_opp], feed_dict={out_labels:out_predictions,
        #                                true_labels:labels})
        # return v[1]

    batches = 0 
    avg_accuracy = 0
    for images, labels in tqdm(validation_generator, desc=data_set):
        batch_accuracy = step(images, labels)
        avg_accuracy += batch_accuracy
        batches += 1
    
    avg_accuracy = avg_accuracy / batches
    print(data_set + " Accruacy:" + str(avg_accuracy))
    return avg_accuracy


def test_single_sample():
    '''
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
        
    model_logits = rgb_model(
        rgb_input, is_training=True, dropout_keep_prob=1.0)
    model_predictions = tf.nn.softmax(model_logits)
    '''
    pass

if __name__ == '__main__':
    main()
