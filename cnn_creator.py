
import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.random.set_seed(2)

from keras import datasets, layers, models
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt


class CNNCreator:
    def __init__(self) -> None:
        self.model = models.Sequential()
        
    
    def build_default_model(self,verbose=0):
        self.model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(Dropout(rate=0.3))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(Dropout(rate=0.3))
        self.model.add(layers.Dense(25, activation = 'softmax'))
        if verbose == 1:
            print(self.model.summary())
        return self.model

    @staticmethod
    def reshape_data(data):
        return  np.array([np.reshape(i, (28, 28)) for i in data])

    def add_layer(type):
        pass

    def get_model(self):
        return self.model


