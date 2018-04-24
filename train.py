import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import pickle
import time
import datetime
import functools
import VLSTM_model as model
import os.path
import gc
import math
from tensorflow.python import debug as tf_debug

#path to traindata
path = 'F:/Dokumente/Programmieren/RoboPen/UnitySimulation/AIRobot_Simulation/DataProcessing/traindata/pre/Data_30x30/diff3_keepLenght14/*'
file = glob.glob(path)
modelName = 'diff3_64LSTM'
data = None
print(file)

#load traindata from the files
for f in file:
    if data is None:
        print("loaded: " + f)
        data = pickle.load(open(f, "rb"))
        gc.collect()
        X = data[0]
        y = data[1]
    else:
        data = pickle.load(open(f, "rb"))
        gc.collect()
        print("loaded: " + f)
        X = np.concatenate((X, data[0]))
        y = np.concatenate((y, data[1]))

#split traindata in test and traiset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

#helpercode to get the mini-batches
def getBatch(courent_batch_position, batch_size, X, y):
    if (courent_batch_position + batch_size) <= X_train.shape[0]:
        nextBatchPosition = (courent_batch_position+batch_size)
        X_batch = X[courent_batch_position : nextBatchPosition,...]
        y_batch = y_train[courent_batch_position : nextBatchPosition,...]
        courent_batch_position += batch_size
    else:
        overLapp = ((courent_batch_position + batch_size) - X_train.shape[0])
        X_batch = X[: overLapp,...]
        y_batch = y[: overLapp,...]
        courent_batch_position = overLapp
    return X_batch, y_batch, courent_batch_position

#helpercode to get the kernals
def put_kernels_on_grid (kernel, grid_Y, grid_X, pad=1):
      '''Visualize conv. features as an image (mostly for the 1st layer).
      Place kernel into a grid, with some paddings between adjacent filters.
      Args:
        kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
        (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                             User is responsible of how to break into two multiples.
        pad:               number of black pixels around each filter (between them)
      Return:
        Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
      '''
      # pad X and Y
      x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

      # X and Y dimensions, w.r.t. padding
      Y = kernel.get_shape()[0] + pad
      X = kernel.get_shape()[1] + pad
      d = kernel.get_shape()[2]

      # put NumKernels to the 1st dimension
      x2 = tf.transpose(x1, (3, 0, 1, 2))
      # organize grid on Y axis
      x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, d]))

      # switch X and Y axes
      x4 = tf.transpose(x3, (0, 2, 1, 3))
      # organize grid on X axis
      x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, d]))

      # back to normal order (not combining with the next step for clarity)
      x6 = tf.transpose(x5, (2, 1, 3, 0))

      # to tf.image_summary order [batch_size, height, width, channels],
      #   where in this case batch_size == 1
      x7 = tf.transpose(x6, (3, 0, 1, 2))

      # scale to [0, 1]
      x_min = tf.reduce_min(x7)
      x_max = tf.reduce_max(x7)
      x8 = (x7 - x_min) / (x_max - x_min)

      return x8

#define parameters
image_width = 30
image_height = 30
image_depth = 3
keep_lenght = 14
num_lable = 8
batch_size = 265
num_epochs = 10000

keep_probability = 0.5
learning_rate_decay = 1e-9
start_learning_rate = 1e-5
beta = 0.5


#define graph
graph = tf.Graph()
with graph.as_default():
    #define placeholders
    X = tf.placeholder(tf.float32, shape=[None, keep_lenght, image_height, image_width, image_depth], name='X')
    X_norm = tf.map_fn(lambda frame1: tf.map_fn(lambda frame2: tf.image.per_image_standardization(frame2), frame1), X)
    y = tf.placeholder(tf.float32, shape=[None, keep_lenght, num_lable], name='y')
    y_cls = tf.argmax(y[:,-1], axis=1)
    #define keep propability for dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(0, trainable=False, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 1500, 0.97, staircase=True)


    pred = model.get_path(image_width, image_height, image_depth, keep_lenght, num_lable, batch_size, X_norm, keep_prob)

    #predictions
    y_pred_cls = tf.argmax(pred, axis=1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y[:,-1]))
    #get l2 regularized value
    regularizer = tf.nn.l2_loss(tf.get_collection('weights')[0]) + tf.nn.l2_loss(tf.get_collection('weights')[1])
    cost_reg = tf.reduce_mean(cost + beta * regularizer)
    #optimize the error function
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_reg, global_step=global_step)
    #get accuracy
    correct_prediction = tf.equal(y_pred_cls, y_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #get incorect classifyed images
    incorrect_images = tf.gather(X, tf.where(tf.not_equal(y_pred_cls, y_cls)))

    #put the kernal on a grid an save it to summary
    grid = put_kernels_on_grid (tf.get_collection('weights')[0][0,...], 4, 2)
    filter_summary = tf.summary.image('test/features', grid, max_outputs=1)
    #save incorect classifyed images to summary
    image_summary = tf.summary.image('test/wronge', incorrect_images[:,0,6,...], max_outputs=20)

    #save training stats to summary
    train_acc = tf.summary.scalar('Accuracy_train', accuracy)
    train_loss = tf.summary.scalar('loss/cost', cost)
    learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)

    #merge and save op
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    #saver for the train and test summary
    train_writer = tf.summary.FileWriter('statistics/{}/train/summ_Modell'.format(modelName), session.graph)
    test_writer = tf.summary.FileWriter('statistics/{}/test/summ_Modell'.format(modelName))

    courent_batch_position = 0
    last_best_acc = 0

    #train loop
    for total_epochs in range(num_epochs):
        #shuffle data after when every image is seen
        if len(X_train)/batch_size % math.ceil(total_epochs+1) == 0:
            X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0, shuffle=True)
        #save summary, save model and print stats to consol
        if total_epochs % 100 == 0:
            #get test summary, courent learning rate und test acc
            summary_train, learning_rate_res, acc_train = session.run([merged, learning_rate, accuracy], feed_dict={X: X_train, y: y_train, keep_prob: 1})
            #wite test summary
            train_writer.add_summary(summary_train, total_epochs)
            #get train summary und train acc
            summary_test, acc_test = session.run([merged, accuracy], feed_dict={X: X_test, y: y_test, keep_prob: 1})
            #write train summary
            test_writer.add_summary(summary_test, total_epochs)
            #print train stats to consol
            print('epoch: {}, Train Acc: {}, Test Acc: {}, Learning rate: {}'.format(total_epochs, acc_train, acc_test, learning_rate_res))
            #save model if it is better then the last saved model
            if last_best_acc < acc_test:
                if not os.path.exists('ckpts/{}/'.format(modelName)):
                    os.makedirs('ckpts/{}/'.format(modelName))
                save_path = saver.save(session, "ckpts/{}/model_acc{}/model_acc{}.ckpt".format(modelName, int(round(acc_test*100)), int(round(acc_test*100))))
                print('Model saved in path: {}'.format(save_path))
                last_best_acc = acc_test

        ###train step###
        #get the courent mini-batch
        X_batch, y_batch, courent_batch_position = getBatch(courent_batch_position, batch_size, X_train, y_train)
        #train model and get the cost value and the l2 reg term
        _, cost_er, reg = session.run([optimizer, cost, regularizer], feed_dict={X: X_batch, y: y_batch, keep_prob: keep_probability})
        print('Epoch: {}, COst: {}, L2 Reg: {}'.format(total_epochs, cost_er, reg), end='\r')
        total_epochs += 1
