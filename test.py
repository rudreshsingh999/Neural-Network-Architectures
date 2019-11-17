import numpy as np
import pandas as pd
import scipy.io
import math
import keras
# from sklearn.cross_validation import train_test_split
# from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Activation, GlobalAveragePooling1D
from keras.layers import *
from keras.models import Sequential

def load_dataset():
	mat_X = scipy.io.loadmat('X_data.mat')['ecg_in_window']
	data_X = np.array(mat_X)
	mat_y = scipy.io.loadmat('y_data.mat')['label']
	data_y = np.array(mat_y)
	data = np.append(data_X,data_y,axis = 1)
	np.random.shuffle(data)

	X = data[0:800 , :-1]
	y = data[0:800 , -1]
	X_test = data[801:1000 , :-1]
	y_test = data[801:1000 , -1]

	# X, y, X_test, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = 42)

	X = np.expand_dims(X,axis=2)
	y = np.expand_dims(y,axis=2)
	X_test = np.expand_dims(X_test,axis=2)
	y_test = np.expand_dims(y_test,axis=2)

	print(X.shape[0])
	print(X.shape)
	return X,y,X_test,y_test

def evaluate_model(X, y, X_test, y_test):
	k = 3
	epochs = 20
	model = Sequential()
	model.add(Convolution1D(filters = 32, kernel_size = k, activation = 'relu', input_shape = (1000,1)))
	# model.add(Convolution1D(filters = 32, kernel_size = k, activation = 'relu'))
	model.add(MaxPooling1D(2))
	model.add(Convolution1D(filters = 64, kernel_size = k, activation = 'relu'))
	model.add(MaxPooling1D(2))
	# model.add(Convolution1D(filters = 64, kernel_size = k, activation = 'relu'))
	model.add(GlobalAveragePooling1D())
	# model.add(Dropout(0.5))
	model.add(Dense(100, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))
	print(model.summary())

	model.compile(loss = 'mse', optimizer = 'rmsprop', metrics = ['accuracy'])
	# fit network
	history = model.fit(X, y, batch_size = 2, epochs = epochs)
	print(history)
	# evaluate network
	loss , accuracy = model.evaluate(X_test, y_test, batch_size = 2) 

	return loss,accuracy

def run_experiment(repeats):
	X, y, X_test, y_test = load_dataset()
	scores = list()
	losss = list()
	for r in range(repeats):
		loss,score = evaluate_model(X, y, X_test, y_test)
		score = score*100.0
		loss = loss*100.0
		scores.append(score)
		losss.append(loss)

	print(scores,losss)

# X, y, X_test, y_test = load_dataset()
# print(evaluate_model(X, y, X_test, y_test))
run_experiment(10)



