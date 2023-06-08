from sklearn.metrics import f1_score
import numpy as np

class CNNEvaluator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def evaluate_model(model_,data_X,data_y,verbose=2):
        data_X = np.array(data_X)
        data_y = np.array(list(map(int, data_y)))
        loss, acc = model_.evaluate(data_X,  data_y, verbose=verbose)
        return loss, acc

    @staticmethod
    def calculate_f1_score(model,data_X,data_y):
        predicted_labels = [np.argmax(i) for i in model.predict(data_X)]
        f1_score_ = f1_score(data_y, predicted_labels, average='weighted')
        return f1_score_



