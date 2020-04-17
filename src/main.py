from data_processing import MSASLDataLoader
import tensorflow as tf

ANNOTATION_FILE_PATH = 'data/MS-ASL/frames/tiny1000.txt'
FRAMES_DIR_PATH = 'data/MS-ASL/frames/global_crops'
from keras.models import Sequential
from keras.layers import Lambda

# Super Naive Approach
def get_model(data_shape):
    model = Sequential()
    # data_shape = (max_frames, color_channels, width, height)
    model.add(Lambda(lambda x: tf.math.reduce_max(x, axis=0), input_shape=data_shape))
    # data_shape = (color_channels, width, height)
    model.add(Lambda(lambda x: tf.math.reduce_max(x, axis=0), input_shape=data_shape[1:]))
    # data_shape = (width, height)
    model.add(Lambda(lambda x: tf.math.reduce_max(x, axis=0), input_shape=data_shape[2:]))
    # data_shape = (height)
    model.add(Lambda(lambda x: tf.math.reduce_sum(x, axis=0), input_shape=data_shape[3:]))
    # data_shape = (1)
    model.add(Lambda(lambda x: tf.math.floormod(tf.math.reduce_sum(x, axis=0), tf.Variable(1000, dtype=tf.keras.backend.dtype(x))), input_shape=data_shape[4:]))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
    
def train(model, train_generator, EPOCHS=1):
    train_info = model.fit_generator(generator=train_generator, epochs=EPOCHS)
    return train_info.history

def main():
    train_generator = MSASLDataLoader(ANNOTATION_FILE_PATH, FRAMES_DIR_PATH, 1, 224, 224, color_mode='rgb', shuffle=True)
    data_shape = train_generator.get_data_dim()
    model = get_model(data_shape)
    train_info = train(model, train_generator)
    print(train_info)

if __name__ == '__main__':
    main()