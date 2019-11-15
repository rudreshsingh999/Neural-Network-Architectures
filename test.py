import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import math


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

X,y,X_test,y_test = load_dataset()


