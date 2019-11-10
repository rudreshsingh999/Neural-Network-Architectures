import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import math

def sigmoid(x):
	r = []
	for i in range(x.size):
		r.append([1 / (1 + math.exp(-x[i]))])
	r = np.asarray(r)
	return r

def gaussian(x,y):
	r = math.exp(-(distance(x,y)**2)/2)
	return r


# Reading Data
mat = scipy.io.loadmat('data5.mat')['x']
data = np.array(mat)

# Feature Scaling
for x in range(len(data[0,:])-1):
	data[:,x] = (data[:,x] - np.mean(data[:,x]))/np.std(data[:,x])

bias = np.expand_dims(np.ones([len(data[:,0])]),axis = 1)
data = np.append(bias,data,axis = 1)

# Cross Validation
np.random.shuffle(data)
X = data[0:1500 , :-1]
X_test = data[1500: , :-1]

y = data[:,-1]
y1 = []

for x in range(len(y)):
	y1.append(1 - y[x])
y = np.expand_dims(y,axis = 1)
y1 = np.expand_dims(y1,axis = 1)
y = np.append(y,y1,axis = 1)

y_test = y[1500: , :] 
y = y[0:1500 , :]

# Implementation
no_of_hidden_neuron = 30
rand = np.random.randn(73,no_of_hidden_neuron)
tempH = np.dot(X,rand)
H = np.tanh(tempH)
w = np.dot(np.linalg.pinv(H),y)

#Testing
tempH_t = np.dot(X_test,rand)
H_t = np.tanh(tempH_t)
y_t = np.dot(H_t,w)

c1=0
c2=0
for x in range(len(y_t)):
	ya = 0
	if(y_t[x][0]>0.5):
		ya = 1
	else:
		ya = 0

	if(ya == y_test[x][0]):
		c1+=1
	else:
		c2+=1

print(c1,c2)

