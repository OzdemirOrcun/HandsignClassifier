

from cnn_creator import CNNCreator
from cnn_evaluator import CNNEvaluator
from cnn_trainer import CNNTRainer
from data_processor import DataProcessor

data_processor = DataProcessor()
train_file_name = 'train_data_label.npz'
test_file_name = 'test_data_label.npz'

train_data, train_label = data_processor.load_train_data(train_file_name)
test_data, test_label = data_processor.load_test_data(test_file_name)

train_data, val_data, train_label, val_label = data_processor.split_data(train_data, train_label, test_size=0.2)

cnn_creator = CNNCreator()

# reshape data
reshaped_train_data = cnn_creator.reshape_data(train_data)
reshaped_test_data = cnn_creator.reshape_data(test_data)
reshaped_val_data = cnn_creator.reshape_data(val_data)


model = cnn_creator.get_model()
model = cnn_creator.build_default_model()


cnn_trainer = CNNTRainer()
model = cnn_trainer.compile_model(model)
history = cnn_trainer.fit_model(model, reshaped_train_data,train_label,reshaped_val_data,val_label)

cnn_evaluator = CNNEvaluator()
test_loss, test_acc = cnn_evaluator.evaluate_model(model,reshaped_test_data,test_label)
print(test_loss, test_acc)


test_f1_score = cnn_evaluator.calculate_f1_score(model,reshaped_test_data,test_label)
print(test_f1_score)
