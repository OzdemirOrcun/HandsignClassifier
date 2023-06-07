import numpy as np
from pymongo import MongoClient


# Connect to MongoDB
client = MongoClient("mongodb+srv://osozdemirorcun:Orcn1997wwfmvpfngo%21@cluster0.sgxsxv7.mongodb.net/")
db = client['data']


# Loading train and test datasets
with np.load('./input/train_data_label.npz') as data:
    train_data = data['train_data']
    train_label = data['train_label']

with np.load('./input/test_data_label.npz') as data:
    test_data = data['test_data']
    test_label = data['test_label']


# Store file IDs in a document in MongoDB
train_collection = db['train']
test_collection = db['test']


for i in range(len(train_data)):
    train_document = {'image': train_data[i].tolist(), 'label': int(train_label[i])}
    train_collection.insert_one(train_document)


for i in range(len(test_data)):
    test_document = {'image': test_data[i].tolist(), 'label': int(test_label[i])}
    test_collection.insert_one(test_document)

# Close the MongoDB connection
client.close()
