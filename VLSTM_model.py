import numpy as np
import tensorflow as tf
import LayerUtills as utills

def get_path(image_width, image_height, image_depth, num_lable, batch_size, data, keep_prob):
    #architecture
    filter_size1 = 11
    num_filters1 = 16
    #maxpool
    filter_size2 = 3
    num_filters2 = 32
    #maxpool

    lstm_size1 = 128

    with tf.name_scope('first_conv_layer_64_filter') as scope:
        layer_conv1, weights_conv1 = utills.create_conv3d_layer(data, 7, image_depth, filter_size1, num_filters1, cnn_stride=2, name='1_conv_layer')
        layer_conv2_pool = utills.pooling3d(layer_conv1, 7, pool_ksize=2, name='layer_1_pooling')
        layer_conv2, weights_conv2 = utills.create_conv3d_layer(layer_conv2_pool,7, num_filters1, filter_size2, num_filters2, cnn_stride=1, name='2_conv_layer')
        layer_conv2_pool = utills.pooling3d(layer_conv2, 7, pool_ksize=2,  name='layer_2_pooling')


    with tf.variable_scope('lstm_layer') as scope:
        layer_flat = utills.flatten_layer3d(layer_conv2_pool, 7, name='flatten_layer')
        layer_lstm = utills.create_lstm_layer(layer_flat, lstm_size1, 7, num_lable, keep_prob, name='1_lstm_cells')

    return layer_lstm
