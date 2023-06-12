
import tensorflow as tf
import numpy as np
np.random.seed(1)
tf.random.set_seed(2)

from keras import datasets, layers, models
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt


class CNNCreator:
    """CNNCrator module creates CNN models for training.
    """    
    def __init__(self) -> None:
        self.model = models.Sequential()
        
    
    def build_default_model(self,verbose=1):
        """Adding default CNN layers to the model

        Parameters
        ----------
        verbose : int, optional
            sets verbose level, by default 1

        Returns
        -------
        object
            CNN model
        """        
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
            return self.model, self.model.summary()
        else:
            return self.model,None

    @staticmethod
    def reshape_data(data):
        """Reshapes data for compiling the model

        Parameters
        ----------
        data : array
            Independent Variables

        Returns
        -------
        numpy array
            Reshaped Independent Variables
        """        
        return  np.array([np.reshape(i, (28, 28)) for i in data])

    def add_layer(type):
        pass

    def get_model(self):
        return self.model


