from sklearn.metrics import f1_score
import numpy as np

class CNNEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate_model(model,data_X,data_y,verbose=2):
        loss, acc = model.evaluate(data_X,  data_y, verbose=verbose)
        return loss, acc

    def calculate_f1_score(model,data_X,data_y):
        predicted_labels = [np.argmax(i) for i in model.predict(data_X)]
        f1_score_ = f1_score(data_y, predicted_labels, average='weighted')
        return f1_score_



