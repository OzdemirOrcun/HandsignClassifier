import numpy as np
from sklearn.model_selection import train_test_split


#file_name = 'train_data_label.npz'

class DataProcessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_train_data(file_name):
        with np.load(file_name) as data:
            train_data = data['train_data']
            train_label = data['train_label']
        return train_data, train_label

    @staticmethod
    def load_test_data(file_name):
        with np.load(file_name) as data:
            test_data = data['test_data']
            test_label = data['test_label']
        return test_data, test_label


    @staticmethod
    def split_data(data_X,data_y,test_size):
        train_data, val_data, train_label, val_label = train_test_split(
            data_X, data_y, test_size=test_size, random_state=54020)
        return train_data, val_data, train_label, val_label

        
