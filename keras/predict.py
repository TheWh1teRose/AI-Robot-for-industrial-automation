import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import glob
import CNN_utils as cnn
import pickle
import time
import datetime
import functools
from PIL import ImageGrab
import cv2
from Controller import Controller
from threading import Thread
import keras

takeTime = 0.3

def updatePositionMatrix():
	global posMatrix
	cntRcv = Controller("127.0.0.1", 5002)
	cntRcv.startController()

	smoothing = 3

	while True:
		UDPData, address = cntRcv.recvData()
		controlsStr = UDPData.decode("utf-8").split("$")[0]
		isRestarted = int(UDPData.decode("utf-8").split("$")[1])
		posInMatrixStr = UDPData.decode("utf-8").split("$")[2]
		posInMatrix = list(map(int, posInMatrixStr.split(":")))
		if posInMatrix[0] < posMatrixSize[0] and posInMatrix[1] < posMatrixSize[1] and posInMatrix[1] < posMatrixSize[1]:
			posMatrix[posInMatrix[0], posInMatrix[1], posInMatrix[2]] = 1





model = keras.models.load_model('ckpts/LRCN_1074_0.05.hdf5')
cnt = Controller("127.0.0.1", 5003)
#cnt.startController()

controlsStr = "0:0:0:0"
posInMatrixStr = "0:0:0"
isRestarted = 0;
posMatrixSize = [60,60,60]
posMatrix = np.zeros((posMatrixSize[0],posMatrixSize[1],posMatrixSize[2]))
lastTime = time.time()

posMatThread = Thread(target = updatePositionMatrix, args = [])
posMatThread.start()

frames = None
lastTime = time.time()
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

	frames = frames[:,1:,...]
	frames = np.concatenate((frames[0,...], data))
	frames = frames[np.newaxis,...]

	frames_min = frames.min(axis=((1,2,3,4)), keepdims=True)
	frames_max = frames.max(axis=((1,2,3,4)), keepdims=True)
	frames = (frames - frames_min)/(frames_max - frames_min)

	prediction = model.predict(frames)
	predicted_data = prediction[0,6,...]
	prediction = np.unravel_index(predicted_data.argmax(), predicted_data.shape)
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
