from data_processing import MSASLDataLoader
import sys
import sonnet as snt
import tensorflow as tf
import numpy as np
import i3d_for_sonnet2 as i3d

print("---------Assert the versions---------")
print("TensorFlow version: {}".format(tf.__version__))
print("    Sonnet version: {}".format(snt.__version__))

# The path containing the information about samples. 
ANNOTATION_FILE_PATH = 'data/MS-ASL/frames/tiny.txt'
ANNOTATION_FILE_PATH_TRAIN = 'data/MS-ASL/frames224/train100.txt'

# The path containing the directories containing each samples frames. 
FRAMES_DIR_PATH = 'data/MS-ASL/frames224/global_crops'

_IMAGE_SIZE = 224
_BATCH_SIZE = 1
_NUM_CLASSES = 100

# Learning rate for the sonnet optimizer
_LEARNING_RATE = 0.01
_NUM_EPOCHS = 200
_MOMENTUM = 0.9

train_generator = MSASLDataLoader(ANNOTATION_FILE_PATH_TRAIN, FRAMES_DIR_PATH, 1, height=224, width=224, color_mode='rgb', shuffle=True, frames_threshold=28)
data_shape = train_generator.get_data_dim()
print('DATA SHAPE (frames per sample, height, width, color_channels)' + str(data_shape))
print('DATA LEN (number of batches)' + str(train_generator.batch_size))
with tf.name_scope('RGB'):
    i3d_model = i3d.InceptionI3d(num_classes=_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    snt.allow_empty_variables(i3d_model)
X_train, y_train = train_generator[0]
print(train_generator[1])
print("Y_Train: ")
print(y_train.shape)
print(y_train)
X_train = tf.cast(X_train, tf.float32)
print('X SHAPE ' + str(X_train.shape))

print("=====================VARIABLES===================")
for variable in tf.compat.v1.global_variables():
    print(variable.name)
    if variable.name.split('/')[0] == 'RGB':
        print(variable.name)
print(tf.compat.v1.get_default_graph().get_name_scope())
print("-------------------------------------------------")

opt = snt.optimizers.SGD(learning_rate=_LEARNING_RATE)

def step(images, labels):
    """Performs one optimizer step on a single mini-batch."""
    with tf.GradientTape() as tape:
        logits, end_points = i3d_model(X_train, is_training=True, dropout_keep_prob=0.3)
        print("Logits--------")
        print(logits.shape)
        print(labels)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
        print(loss)

    print("=====================VARIABLES===================")
    print(tf.compat.v1.get_default_graph().get_name_scope())
    print("-------------------------------------------------")
    params = i3d_model.trainable_variables
    print("Loss-----")
    print(loss.shape)
    print("Params: {}".format(params))
    grad = tape.gradient(loss, params)
    opt.apply(grad, params)
    return loss

print(step(X_train, y_train))