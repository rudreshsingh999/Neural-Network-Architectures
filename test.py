import numpy as np
import pandas as pd
import scipy.io
import math
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential

def load_dataset():
	mat_X = scipy.io.loadmat('X_data.mat')['ecg_in_window']
	data_X = np.array(mat_X)
	print(data_X.shape)
	mat_y = scipy.io.loadmat('y_data.mat')['label']
	data_y = np.array(mat_y)
	data = np.append(data_X,data_y,axis = 1)
	np.random.shuffle(data)

	X = data[0:800 , :-1]
	y = data[0:800 , -1]
	X_test = data[801:1000 , :-1]
	y_test = data[801:1000 , -1]
	print(X.shape[0])

	return X,y,X_test,y_test

def evaluate_model(X, y, X_test, y_test):
	k = 41
	epochs = 10
	model = Sequential()
	# model.add(Reshape((1000,1),input_shape=(1000,)))
	model.add(Convolution1D(filters = 1, kernel_size = k, activation = 'relu', input_shape = (1000,1)))
	# model.add(GlobalAveragePooling1D())
	model.add(Dense(100, activation = 'softmax'))
	model.add(Dense(2, activation = 'softmax'))
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])

	# fit network
	model.fit(X, y, epochs = epochs, batch_size = 800, verbose = 0)

	# evaluate network
	_, accuracy = model.evaluate(X_test, y_test, batch_size = 200, verbose = 0) 

	return accuracy

def run_experiment(repeats):
	X, y, X_test, y_test = load_dataset()
	scores = list()

	for r in range(repeats):
		score = evaluate_model(X, y, X_test, y_test)
		score = score*100.0
		scores.append(score)

	print(scores)

run_experiment(10)



