import time
import xgboost as xgb
from sklearn.svm import  LinearSVC as SVC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.ensemble import  AdaBoostClassifier as ABC,  RandomForestClassifier as RFC


# Base class of ML models
class ML:
    
    def __init__(self, model):
        self.model = model

    def predict(self, x_test):
        return self.model.predict(x_test)

    def fit(self, x_train, y_train):
        print(f'Start fitting {type(self.model).__name__} model, shape of x_train: {x_train.shape}, shape of y_train: {y_train.shape}')
        t = time.time()
        self.model.fit(x_train, y_train)
        self.train_time = (time.time() - t)/60
        print(f'{type(self.model).__name__} trained in {round(self.train_time, 2)} minutes')

    def get_train_time(self):
        return self.train_time

# Class of each ML model

class XGBClassifier(ML):
    def __init__(self, params):
        super().__init__(xgb.XGBClassifier(**params))

class LogisticRegression(ML):
    def __init__(self, params):
        super().__init__(LogReg(**params))


class LinearSVC(ML):
    def __init__(self, params):
        super().__init__(SVC(**params))


class AdaBoostClassifier(ML):
    def __init__(self, params):
        super().__init__(ABC(**params))


class RandomForestClassifier(ML):
    def __init__(self, params):
        super().__init__(RFC(**params))
