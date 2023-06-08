from data_processor import DataProcessor
dp = DataProcessor()

train_data,train_label = dp.load_train_data()

print(len(train_data),len(train_label))
print('done')

test_data,test_label = dp.load_test_data()
print(len(test_data),len(test_label))
print('done')