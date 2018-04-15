import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import glob
import pickle
import time
import datetime
import functools
from PIL import ImageGrab
import cv2
import VLSTM_model as model
from Controller import Controller
from threading import Thread

#time per frame
takeTime = 0.3

#function for the comunicaton thread with unity
def update():
	#start controller to unity
	cntRcv = Controller("127.0.0.1", 5002)
	cntRcv.startController()

	#get the data from unity
	while True:
		UDPData, address = cntRcv.recvData()
		#get the pressed keys from unity
		controlsStr = UDPData.decode("utf-8").split("$")[0]
		#returns 1 if the envierement is reseted
		isRestarted = int(UDPData.decode("utf-8").split("$")[1])

#parameters
image_width = 30
image_height = 30
image_depth = 3
state_length = 7
num_lable = 8

tf.reset_default_graph()

#generate graph
graph = tf.Graph()
with graph.as_default():
	#define placeholders
	X = tf.placeholder(tf.float32, shape=[None, state_length, image_height, image_width, image_depth], name='X')
	X_norm = tf.map_fn(lambda frame1: tf.map_fn(lambda frame2: tf.image.per_image_standardization(frame2), frame1), X)

	#make the prediction
	pred = model.get_path(image_width, image_height, image_depth, num_lable, 1, X_norm, 1)
	pred_softmax = tf.nn.softmax(pred)
	y_pred_cls = tf.argmax(pred_softmax, axis=1)

	saver = tf.train.Saver()

#start the controller to send the controlls to unity
cnt = Controller("127.0.0.1", 5003)

controlsStr = "0:0:0:0"
isRestarted = 0;
lastTime = time.time()

#start the update thread for the communication with unity
updateThread = Thread(target = update, args = [])
updateThread.start()

frames = None
lastTime = time.time()
with tf.Session(graph=graph) as sess:
	#restore model
	saver.restore(sess, "ckpts/model_acc91/model_acc91.ckpt")
	while True:
		#wait until the take time is over
		yet = time.time()
		while (yet-lastTime) < takeTime:
			yet = time.time()
		lastTime = time.time()

		cls_predection = None
		data = None
		#get the image from the screen
		printscreen = np.array(ImageGrab.grab(bbox=(2,50,302,350)))
		#downsize image
		printscreen = cv2.resize(printscreen, dsize=(30, 30), interpolation=cv2.INTER_CUBIC)

		#shape: [1,30,30,3]
		data = printscreen[np.newaxis,...]
		#show image
		cv2.imshow('screen', np.array(data[0,...],dtype=np.int8))

		#initialize frame
		if frames is None:
			frames = np.zeros((1,7,30,30,3), dtype=np.float32)

		#gemerate frame series
		#delete last frame
		#shape: [1,6,30,30,3]
		frames = frames[:,1:,...]
		#concate data to the frame series
		#shape: [7,30,30,3]
		frames = np.concatenate((frames[0,...], data))
		#add new axis
		#shape: [1,7,30,30,3]
		frames = frames[np.newaxis,...]

		#make prediction
		prediction, softmax = sess.run([y_pred_cls, pred_softmax], feed_dict={X:frames})
		#coverted to one hot labels
		cls_predection = np.zeros(8, dtype=np.int8)
		cls_predection[prediction] = 1

		#send data to unity
		send_data = ''
		for p in cls_predection:
			send_data = send_data + str(p) + ':'
		send_data = send_data[:-1]
		print(send_data)
		cnt.sendMessage(send_data.encode('utf-8'))
		if cv2.waitKey(25) & 0xFF == ord('q'): #quit statement
			cv2.destroyAllWindows()
			break
