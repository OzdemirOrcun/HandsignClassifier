import numpy as np
from sklearn.model_selection import train_test_split

import pymongo

#file_name = 'train_data_label.npz'

class DataProcessor:
    def __init__(self) -> None:

        # Create a MongoDB client
        self.client = pymongo.MongoClient("mongodb+srv://osozdemirorcun:Orcn1997wwfmvpfngo%21@cluster0.sgxsxv7.mongodb.net/")
        self.db = self.client['data']
        self.train_collection = self.db['train']
        self.test_collection = self.db['test']

    def load_train_data(self,file_name=None,local=None):

        # Retrieve all documents from the collection
        documents = self.train_collection.find()
        train_data = []
        train_label = []
        # Process the retrieved documents
        for document in documents:
            # Access fields within the document
            image = document["image"]
            label = document["label"]
            train_data.append(image)
            train_label.append(label)

        if local:
            with np.load(file_name) as data:
                train_data = data['train_data']
                train_label = data['train_label']
        return train_data, train_label

    def load_test_data(self,file_name=None,local=None):
        # Retrieve all documents from the collection
        documents = self.test_collection.find()
        test_data = []
        test_label = []
        # Process the retrieved documents
        for document in documents:
            # Access fields within the document
            image = document["image"]
            label = document["label"]
            test_data.append(image)
            test_label.append(label)

        if local:
            with np.load(file_name) as data:
                test_data = data['test_data']
                test_label = data['test_label']
        return test_data, test_label


    @staticmethod
    def split_data(data_X,data_y,test_size):
        train_data, val_data, train_label, val_label = train_test_split(
            data_X, data_y, test_size=test_size, random_state=54020)
        return train_data, val_data, train_label, val_label

        

    @staticmethod
    def reshape_data(data):
        return  np.array([np.reshape(i, (28, 28)) for i in data])
