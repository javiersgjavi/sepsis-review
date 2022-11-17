import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

class MetricCalculator:
    def __init__(self):
        self.metrics = ['auprc', 'roc_auc', 'accuracy']

    def get_metrics(self, real, prediction):
        results = []

        for metric in self.metrics:
            value = getattr(self, metric)(real, prediction)
            results.append(value)
            
        return results

    def get_name_metrics(self):
        return self.metrics


    def roc_auc(self, real, prediction):
        res = roc_auc_score(real, prediction)
        return res

    def auprc(self, real, prediction):
        return average_precision_score(real, prediction)

    def accuracy(self, real, prediction):
        prediction = np.where(prediction>0.5, 1, 0)
        return accuracy_score(real, prediction)
