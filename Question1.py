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
y = data[0:1500 , -1]
y_test = data[1500: , -1]


d = len(X[0,:])
h = 9
i = 2
w = np.random.random((h,d))
w = np.subtract(np.multiply(w,2),1)
v = np.random.random((i,h))
v = np.subtract(np.multiply(v,2),1)

# zh = np.dot(w,X[0,:])
# zh = sigmoid(zh)

# yi = np.dot(v,zh)
# yi = sigmoid(yi)
# print(yi)

# a = np.arange(4).reshape(2,2)
# a[0][0] = 1
# print(a)
rep=0
alpha = 0.001
while (rep < 1):
	for t in range(len(X[:,0])):
		zh = np.dot(w,X[t,:])
		zh = sigmoid(zh)

		yi = np.dot(v,zh)
		yi = sigmoid(yi)
		delta_v = np.arange(i*h).reshape(i,h)
		delta_w = np.arange(h*d).reshape(h,d)
		for k in range(i):
			for l in range(h):
				delta_v[k][l] = -1*alpha*(abs(k-y[t]) - yi[k])*yi[k]*(1-yi[k])*zh[l]

		for l in range(h):
			for j in range(d):
				for k in range(i):
					q = (abs(k-y[t]) - yi[k])*v[k][l]
				delta_w[l][j] = -1*alpha*q*zh[l]*(1-zh[l])*X[t][j]

		v = np.subtract(v,delta_v)
		w = np.subtract(w,delta_w)
	rep+=1

print(w,v)

# testing
c1=0
c2=0
c3=0
for t in range(len(y_test)):
	z1 = np.dot(w,X_test[t,:])
	z1 = sigmoid(z1)
	y1 = np.dot(v,z1)
	y1 = sigmoid(y1)
	if(y[t]==1):
		c3+=1
	if(y1[0]>0.5):
		ya = 1
	else:
		ya = 0

	print(ya)
	if(ya == y[t]):
		c1+=1
	else:
		c2+=1

print(c1,c2,c3)
