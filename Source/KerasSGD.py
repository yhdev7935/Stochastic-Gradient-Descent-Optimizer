from sklearn.datasets import make_regression
import numpy as np
import tensorflow as tf

class KerasSGD:
    
    def __init__(self, n_samples, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.X, self.Y = self.DataGeneration()
        self.Y = np.expand_dims(self.Y, axis = 1)
        self.model = self.modelGeneration()
        
    def DataGeneration(self):
        return make_regression(n_samples = self.n_samples,
                                n_features = 1,
                                bias = 10.0,
                                noise = 10.0, random_state = 2)
    
    def modelGeneration(self):
        
        # variables definition
        model = tf.keras.Sequential()
        linear = tf.keras.layers.Dense(1, activation = 'linear')
        optimizer = tf.keras.optimizers.SGD(lr = self.learning_rate)
        
        model.add(linear)
        model.compile(loss = 'mse',
                     optimizer = optimizer,
                     metrices = ['mse'])
        
        return model
    
    def Fitting(self, maxsteps):
        self.X_test = self.X[:int(self.n_samples / 4 + 1)]
        self.Y_test = self.Y[:int(self.n_samples / 4 + 1)]
        self.X_train = self.X[int(self.n_samples / 4 + 1):]
        self.Y_train = self.Y[int(self.n_samples / 4 + 1):]
        return self.model.fit(self.X_train, self.Y_train, batch_size = 10,
                      epochs = maxsteps, shuffle = True, validation_data = (self.X_test, self.Y_test))
        
    def evaluateTestData(self):
        return self.model.evaluate(self.X_test, self.Y_test)
