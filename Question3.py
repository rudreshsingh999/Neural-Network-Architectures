import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()


class NeuronLayer:
    def __init__(self, n, input_len):

        self.bias = np.random.rand(n, 1)
        self.weights = np.random.rand(n, input_len)
        

    def sigmoid(self, a):
        return 1/(1 + np.exp(-a))

    def feed_forward(self, input_):
        self.z = np.dot(input_, self.weights.T)
        self.a = self.sigmoid(self.z)
        return self.a


 


class Autoencoder:
        
    def __init__(self, input_size, hidden_size, output_size):
        self.alpha = 0.1
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.hidden = NeuronLayer(hidden_size, input_size)
        self.output = NeuronLayer(output_size, hidden_size)
        
    def feed_forward(self, input_):
        hidden_layer_outputs = self.hidden.feed_forward(input_)
        self.output.feed_forward(hidden_layer_outputs)
        

    def error_y(self, actual, predicted):
        return np.subtract(predicted, actual)

    def y_z(self, predicted):
        return predicted * (1 - predicted)

    def error_deriv(self, actual, predicted):
        s = np.multiply(np.subtract(predicted, actual), predicted * (1 - predicted))
        return s


    #training function with backprop
    def train(self, X_train, y_train):
        self.feed_forward(X_train)
        #calculating deltas
        deltas_output = self.error_deriv(y_train, self.output.a)
        deltas_hidden = np.dot(deltas_output, self.hidden.weights.T)
        deltas_hidden = np.multiply(deltas_hidden, self.y_z(self.hidden.a))
        delerror_delw = (1/X_train.shape[0])*np.dot(deltas_output.T,self.hidden.a)
        self.output.weights -= self.alpha*(delerror_delw)
        self.output.bias -= (1/X_train.shape[0])*self.alpha*(np.sum(deltas_output))
        delerror_delw = (1/X_train.shape[0])*np.dot(deltas_hidden.T, X_train)
        self.hidden.weights -= self.alpha*(delerror_delw)
        self.hidden.bias -= (1/X_train.shape[0])*self.alpha*(np.sum(deltas_hidden))
        
        error = y_train - self.output.a
        error = np.square(error)
        error = np.sum(error)
        error = (1/y_train.shape[0])*np.sqrt(error)
        print(error)
        
    def get_hidden(self):
        return self.hidden


class FineTune:
    
    def __init__(self, output_size, hidden_size):
        self.alpha = 0.1
        self.output = NeuronLayer(output_size, hidden_size)
        
    def feed_forward(self, input_):
        self.output.feed_forward(input_)

    def error_y(self, actual, predicted):
        return np.subtract(predicted, actual)

    def y_z(self, predicted):
        return predicted * (1 - predicted)

    def error_deriv(self, actual, predicted):
        s = np.multiply(np.subtract(predicted, actual), predicted * (1 - predicted))
        return s


    #training function with backprop
    def train(self, X_train, y_train):
        self.feed_forward(X_train)
        
        #calculating deltas
        y_train = np.reshape(y_train,[y_train.shape[0],1])
        deltas_output = self.error_deriv(y_train, self.output.a)
        delerror_delw = (1/X_train.shape[0])*np.dot(deltas_output.T, X_train)
        self.output.weights -= self.alpha*(delerror_delw)
        self.output.bias -= (1/X_train.shape[0])*self.alpha*(np.sum(deltas_output))
        
        error = y_train - self.output.a
        error = np.square(error)
        error = np.sum(error)
        error = (1/y_train.shape[0])*np.sqrt(error)
        print(error)
        
    def predict(self, X_test, y_test):
        y_pred = np.dot(X_test, self.output.weights.T)
        y_pred = normalizer.fit_transform(y_pred)
        preds = []
        for i in range(len(y_test)):
            if y_pred[i] <= 0:
                preds.append(0)
            else:
                preds.append(1)
        print(preds)
        print("Accuracy ->",accuracy_score(y_test, preds))
        
    

import scipy.io
mat = scipy.io.loadmat('data5.mat')


X = mat['x'][:,[i for i in range(72)]]
y = mat['x'][:,72]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

size_1 = 72
size_2 = 40
size_3 = 30
size_4 = 15
epochs = 250

autoencoder1 = Autoencoder(size_1,size_2,size_1)
autoencoder2 = Autoencoder(size_2,size_3,size_2)
autoencoder3 = Autoencoder(size_3,size_4,size_3)



print("Pre-training Autoencoder 1:")
for i in range(1,epochs):
    print("Epoch " + str(i))
    autoencoder1.train(X_train, X_train)

print("Pre-training Autoencoder 2:")
for i in range(1,epochs):
    print("Epoch " + str(i))
    autoencoder2.train(autoencoder1.hidden.a,autoencoder1.hidden.a)

print("Pre-training Autoencoder 3:")
for i in range(1,epochs):
    print("Epoch " + str(i))
    autoencoder3.train(autoencoder2.hidden.a,autoencoder2.hidden.a)
    
finetuner = FineTune(1, size_4)
for i in range(1,epochs):
    print("Epoch " + str(i))
    finetuner.train(autoencoder3.hidden.a, y_train)
    
autoencoder1.feed_forward(X_test)
autoencoder2.feed_forward(autoencoder1.hidden.a)
autoencoder3.feed_forward(autoencoder2.hidden.a)

finetuner.predict(autoencoder3.hidden.a,y_test)