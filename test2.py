import numpy as np
import pandas as pd
import scipy.io
import math
import keras
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
# from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Activation, GlobalAveragePooling1D
import keras.backend as K
from keras.layers import *
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import RMSprop
import gzip
from matplotlib import pyplot as plt


def load_dataset():
    mat_X = scipy.io.loadmat('X_data.mat')['ecg_in_window']
    X = np.array(mat_X)
    np.random.shuffle(X)
    # X, y, X_test, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = 42)
    X = np.expand_dims(X,axis=2)
    print(X.shape[0])
    print(X.shape)
    return X

def autoencoder(input_X):
    #encoder
    print(input_X.shape)
    conv1 = Conv2D(32, (5, 1), activation='relu', padding='same')(input_X)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (5, 1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (5, 1), activation='relu', padding='same')(pool2) 

    #decoder
    conv4 = Conv2D(128, (5, 1), activation='relu', padding='same')(conv3)
    up1 = UpSampling2D((2,2))(conv4)
    conv5 = Conv2D(64, (5, 1), activation='relu', padding='same')(conv2) 
    up2 = UpSampling2D((2,2))(conv5)
    decoded = Conv2D(1, (5, 1), activation='relu', padding='same')(up2) 
    return decoded



# train_data = X.reshape(-1, 100,10, 1)
X = Input(shape = (100, 10, 1))
autoencoder = Model(X, autoencoder(X))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
print(autoencoder.summary())

X = load_dataset()
train_data = X.reshape(1000,100,10,1)
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,train_data,test_size = 0.2,random_state = 13)
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=2,epochs=50,verbose=1,validation_data=(valid_X, valid_ground))

print(train_X.shape)

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(50)
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training loss')
plt.legend()
plt.show()


