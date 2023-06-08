
from tabnanny import verbose
from time import time
import tensorflow as tf

#TODO: hyperparameter tuning
#TODO: weight download, model save

class CNNTrainer:
    def __init__(self) -> None:
        self.learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

        ## compile hyperparameters
        self.optimizer = 'adam'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = ["accuracy"]

        ## fit hyperparameters
        self.epochs = 3
        self.batch_size = 100
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
        self.learning_rate_reduction]


    def compile_model(self,model):
        model.compile(optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=self.metrics)
        return model


    def fit_model(self,model,train_X,train_y,test_X,test_y):
        start = time()
        history = model.fit(train_X, train_y, epochs=self.epochs, batch_size = self.batch_size,   
                            validation_data=(test_X, test_y),callbacks=self.callbacks)
        print(f'Time taken to run: {time() - start} seconds')
        return history