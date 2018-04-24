import numpy as np
import tensorflow as tf
import CNN_utils as cnn
import keras

def VGG_A(keep_prob, split_size):
    #architecture
    filter_size1 = 11
    num_filters1 = 16
    filter_size2 = 4
    num_filters2 = 32

    rnn_size1 = 128

    image_width = 30
    image_height = 30
    image_depth = 3
    num_lable = 8
    batch_size = 128

    imput_shape = [7, image_width, image_height, image_depth]

    model = keras.models.Sequential()

    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters1, (filter_size1, filter_size1), strides=(2,2), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((3, 3), strides=(2, 2))))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.Conv2D(num_filters2, (filter_size2, filter_size2), strides=(1,1), activation='relu', padding='same'), input_shape=imput_shape))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.convolutional.MaxPooling2D((3, 3), strides=(2, 2))))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.Flatten()))

    model.add(keras.layers.recurrent.LSTM(rnn_size1, return_sequences=True, dropout=keep_prob))
    model.add(keras.layers.Dropout(keep_prob))
    model.add(keras.layers.wrappers.TimeDistributed(keras.layers.Dense(num_lable, activation='softmax')))
    print(model.summary())

    return model
