from sklearn.metrics import f1_score
import numpy as np

class CNNEvaluator:
    """CNNEvaluator module loads CNN models and applies evaluation metrics to them.
    """    
    def __init__(self) -> None:
        pass

    @staticmethod
    def evaluate_model(model_,data_X,data_y,verbose=2):
        """Uses keras' default evaluation function

        Parameters
        ----------
        model_ : object
            Trained and saved CNN Model to be evaluated.
        data_X : numpy array
            Independent Variables to fit into evaluation function.
        data_y : numpy array
            Dependent Variables to fit into evaluation function.
        verbose : int, optional
            Sets verbose levels, by default 2

        Returns
        -------
        tuple
            loss and accuracy results of the evaluated CNN model.
        """        
        data_X = np.array(data_X)
        data_y = np.array(list(map(int, data_y)))
        loss, acc = model_.evaluate(data_X,  data_y, verbose=verbose)
        return loss, acc

    @staticmethod
    def calculate_f1_score(model,data_X,data_y):
        """_summary_

        Parameters
        ----------
        model : object
            Trained and saved CNN Model to be evaluated.
        data_X : array
            Independent Variables to fit into evaluation function.
        data_y : array
            Dependent Variables to fit into evaluation function.

        Returns
        -------
        float
            F1 score of evaluated CNN model.
        """        
        predicted_labels = [np.argmax(i) for i in model.predict(data_X)]
        f1_score_ = f1_score(data_y, predicted_labels, average='weighted')
        return f1_score_



