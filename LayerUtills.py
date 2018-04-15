import numpy as np
import tensorflow as tf

#create weights with truncated_normal function
def create_weights(shape, name=None):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

#create biases mith constant
def create_biases(lenght, name=None):
	return tf.Variable(tf.constant(0.05, shape=[lenght]))

#create an convolutional layer
def create_conv_layer(input, num_input_channels, filter_size, num_filters, name=None, cnn_stride=1):
	with tf.variable_scope(name) as scope:
		shape = [filter_size, filter_size, num_input_channels, num_filters]

		weights_name = 'weights ' + name
		weights = create_weights(shape, weights_name)

		biases_name = 'biases ' + name
		biases = create_biases(num_filters, biases_name)

		layer_name = 'conv2d_' + name
		layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, cnn_stride, cnn_stride, 1], padding='SAME', name=layer_name)
		layer += biases

		relu_name = 'relu_' + name
		layer = tf.nn.relu(layer, name=relu_name)

		return layer, weights

#create an 3d convolutional layer
def create_conv3d_layer(input, batch, num_input_channels, filter_size, num_filters, name=None, cnn_stride=1):
	with tf.variable_scope(name) as scope:
		shape = [batch, filter_size, filter_size, num_input_channels, num_filters]

		weights_name = 'weights ' + name
		weights = create_weights(shape, weights_name)

		biases_name = 'biases ' + name
		biases = create_biases(num_filters, biases_name)

		tf.add_to_collection('weights', weights)

		layer_name = 'conv3d_' + name
		layer = tf.nn.conv3d(input=input, filter=weights, strides=[1, 1, cnn_stride, cnn_stride, 1], padding='SAME', name=layer_name)
		layer += biases

		relu_name = 'relu_' + name
		layer = tf.nn.relu(layer, name=relu_name)

		return layer, weights

def to_sequence(input, batch):
	output = []
	for i in range(batch):
		output.append(input[:,i,...])
	return output

#create an fully connected layer
def create_fully_connected_layer(input, num_inputs, num_outputs, relu=True, name=None):
	with tf.variable_scope(name) as scope:
		weights_name = 'weights ' + name
		weights = create_weights([num_inputs, num_outputs], name=weights_name)
		biases_name = 'biases ' + name
		biases = create_biases(num_outputs, name=biases_name)

		layer = tf.matmul(input, weights) + biases

		if relu:
			relu_name = 'relu_' + name
			layer = tf.nn.relu(layer, name=relu_name)

		return layer

#create an LSTM Layer
def create_lstm_layer(input, num_cells, batch, num_outputs, keep_prob, name=None):
	with tf.variable_scope(name) as scope:
		#create weights and biases
		weights_name = 'weights ' + name
		weights = create_weights([num_cells, num_outputs], name=weights_name)
		biases_name = 'biases ' + name
		biases = create_biases(num_outputs, name=biases_name)

		tf.add_to_collection('weights', weights)

		#create cell
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_cells)
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
		#generate output
		outputs, states = tf.nn.dynamic_rnn(lstm_cell, input[0], dtype=tf.float32)
		layer = tf.matmul(outputs[:,-1], weights) + biases
		return layer


#function for local response normalization
def local_response_normalization(input, name=None):
	return tf.nn.local_response_normalization(input, name=name)

#dropout function
def dropout(input, keep_prob=0.5, name=None):
	return tf.nn.dropout(input, keep_prob, name=name)

#max-pooling function
def pooling(input, pool_ksize=2, pool_stride=2, name=None):
	return tf.nn.max_pool(input, [1, pool_ksize, pool_ksize, 1], [1, pool_stride, pool_stride, 1], padding='SAME', name=name)

#create 3d pooling layer
def pooling3d(input, batch, pool_ksize=2, pool_stride=2, name=None):
	return tf.nn.max_pool3d(input, [1, 1, pool_ksize, pool_ksize, 1], [1, 1, pool_stride, pool_stride, 1], padding='SAME', name=name)

#flatt the output of an 3d convolutional layer
def flatten_layer3d(layer, batch, name=None):
	with tf.variable_scope(name) as scope:
		layer_shape = layer.get_shape()
		num_features = layer_shape[2:5].num_elements()
		layer_flat = tf.reshape(layer, [-1, batch, num_features])
		print(layer_flat)
		return layer_flat, num_features

#flatten input data
def flatten_layer(layer, name=None):
	with tf.variable_scope(name) as scope:
		layer_shape = layer.get_shape()
		num_features = layer_shape[1:4].num_elements()
		layer_flat = tf.reshape(layer, [-1, num_features])
		return layer_flat, num_features
