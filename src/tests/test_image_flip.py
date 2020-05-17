import keras
import numpy as np

path = "data/MS-ASL/frames224/global_crops/__lLQ3mhCvM/clip_0000/img_00001.jpg"

img = keras.preprocessing.image.load_img(
            path
)

x_ = keras.preprocessing.image.img_to_array(img, data_format='channels_last')

keras.preprocessing.image.save_img(
    "src/tests/images/test.jpg", x_, data_format=None, file_format=None, scale=True
)

x_ = np.flip(x_, 1)

keras.preprocessing.image.save_img(
    "src/tests/images/flip_test.jpg", x_, data_format=None, file_format=None, scale=True
)