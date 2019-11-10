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

def distance(c,c_temp):
	r = 0
	for x in range(len(c)):
		r = r + (c[x]-c_temp[x])**2
	return math.sqrt(r)

def k_means(data,k):
	centers = []
	for x in range(k):
		centers.append(data[x])
	centers = np.array(centers)
	centers_temp = np.zeros((k,73))

	exp_error = 0.0001
	p_error = 0

	for x in range(len(centers)):
		p_error = p_error + distance(centers[x],centers_temp[x])

	while(p_error > exp_error):
		clusters = [0]*k
		count = [0]*k
		for x in range(len(data)):
			dist = np.zeros(k)
			for y in range(len(centers)):
				dist[y] = distance(data[x],centers[y])

			mini = 100000000000
			i = -1
			for y in range(len(dist)):
				if(dist[y] < mini):
					mini = dist[y]
					i = y

			clusters[i] = np.add(clusters[i],data[x])
			count[i] = count[i] + 1

		for x in range(len(centers)):
			centers_temp[x] = clusters[x]/count[x]

		p_error = 0
		for x in range(len(centers)):
			p_error = p_error + distance(centers[x],centers_temp[x])

		centers = centers_temp

	return centers



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
k = 20
centers = k_means(X,k)
centers = np.array(centers)


H = np.random.random((len(X),k))
for x in range(len(X)):
	for y1 in range(k):
		H[x][y1] = gaussian(X[x],centers[y1])

bias = np.expand_dims(np.ones(1500),axis = 1)
H = np.append(bias,H,axis = 1)
H = np.linalg.pinv(H)
w = np.dot(H,y)
print(w)

# Testing
H1 = np.random.random((len(X_test),k))
for x in range(len(X_test)):
	for y1 in range(k):
		H1[x][y1] = gaussian(X_test[x],centers[y1])

bias = np.expand_dims(np.ones(len(X_test)),axis = 1)
H1 = np.append(bias,H1,axis = 1)
yp = np.dot(H1,w)

c1=0
c2=0
print(yp)
for x in range(len(yp)):
	ya = 0
	if(yp[x][0]>0.5):
		ya = 1
	else:
		ya = 0

	if(ya == y_test[x][0]):
		c1+=1
	else:
		c2+=1

print(c1,c2)