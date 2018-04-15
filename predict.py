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

takeTime = 0.3

def update():
	cntRcv = Controller("127.0.0.1", 5002)
	cntRcv.startController()

	while True:
		UDPData, address = cntRcv.recvData()
		controlsStr = UDPData.decode("utf-8").split("$")[0]
		isRestarted = int(UDPData.decode("utf-8").split("$")[1])

image_width = 30
image_height = 30
image_depth = 3
state_length = 7
num_lable = 8
batch_size = 265

tf.reset_default_graph()

graph = tf.Graph()
with graph.as_default():
	#define placeholders
	X = tf.placeholder(tf.float32, shape=[None, state_length, image_height, image_width, image_depth], name='X')
	#define keep propability for dropout

	pred = model.get_path(image_width, image_height, image_depth, num_lable, batch_size, X, 1)
	pred_softmax = tf.nn.softmax(pred)
	y_pred_cls = tf.argmax(pred_softmax, axis=1)
	#predictions
	saver = tf.train.Saver()


cnt = Controller("127.0.0.1", 5003)
#cnt.startController()

controlsStr = "0:0:0:0"
isRestarted = 0;
lastTime = time.time()

updateThread = Thread(target = update, args = [])
updateThread.start()

frames = None
lastTime = time.time()
with tf.Session(graph=graph) as sess:
	saver.restore(sess, "ckpts/model_acc91/model_acc91.ckpt")
	while True:
		yet = time.time()
		while (yet-lastTime) < takeTime:
			yet = time.time()
		lastTime = time.time()
		data = None
		cls_predection = None
		printscreen = np.array(ImageGrab.grab(bbox=(2,50,302,350)))
		printscreen = cv2.resize(printscreen, dsize=(30, 30), interpolation=cv2.INTER_CUBIC)

		x = np.stack((printscreen[:,:,0], printscreen[:,:,1], printscreen[:,:,2]), axis=2)

		data = x[np.newaxis,...]
		cv2.imshow('screen', np.array(data[0,:,:,:3],dtype=np.int8))

		if frames is None:
			frames = np.zeros((1,7,30,30,3), dtype=np.float32)
			for i in range(7):
				frames[0,i,...] = data

		frames = frames[:,:-1,...]
		frames = np.concatenate((frames[0,...], data))
		frames = frames[np.newaxis,...]

		frames_min = frames.min(axis=((1,2,3,4)), keepdims=True)
		frames_max = frames.max(axis=((1,2,3,4)), keepdims=True)
		frames = (frames - frames_min)/(frames_max - frames_min)

		prediction, softmax = sess.run([y_pred_cls, pred_softmax], feed_dict={X:frames})
		print(softmax)
		print(prediction)
		cls_predection = np.zeros(8, dtype=np.int8)
		cls_predection[prediction] = 1

		send_data = ''
		for p in cls_predection:
			send_data = send_data + str(p) + ':'
		send_data = send_data[:-1]
		print(send_data)
		cnt.sendMessage(send_data.encode('utf-8'))
		time.sleep(0.13)
		if cv2.waitKey(25) & 0xFF == ord('q'): #quit statement
			cv2.destroyAllWindows()
			break
