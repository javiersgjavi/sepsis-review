import os
import sys
import pickle
import inspect
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

from classes.Data import Data, ParamsGrid, DataChallenge
from classes.Metrics import MetricCalculator
from classes.DL import GRU, TCN, CNN, LSTM, MLP
from classes.ML import LinearSVC, XGBClassifier, LogisticRegression, AdaBoostClassifier, RandomForestClassifier
from utils.preprocess_data import preprocess_general

class ParameterOptimization:
    def __init__(self, data, name, models, imputation_methods, iterations_sampler):

        self.data_source = data

        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.name = name
        self.models_to_create = models
        self.iter_sampler = iterations_sampler
        self.data_original_path='./data/original/'

        self.models = {
            'GRU': GRU,
            'TCN': TCN,
            'CNN': CNN,
            'MLP': MLP,
            'LSTM': LSTM,
            'LinearSVC': LinearSVC,
            'XGBClassifier': XGBClassifier,
            'LogisticRegression': LogisticRegression,
            'AdaBoostClassifier' : AdaBoostClassifier,
            'RandomForestClassifier': RandomForestClassifier
            
        }

        self.metric_calculator = MetricCalculator()

        self.imputation_methods = []
        for method in imputation_methods.keys():
            if imputation_methods[method] == 1:
                self.imputation_methods.append(method)

        self.dl_models = [cls_name for cls_name, cls_obj in inspect.getmembers(sys.modules['classes.DL']) if inspect.isclass(cls_obj)]

    def _load_data(self, imputation_method, norm_method):
        if self.data_source == 'MIMIC-III':
            print(f'Loading data with imputation method: {imputation_method}')

            with open(f'{self.data_original_path}train_data.pkl', 'rb') as f:
                train_data = pickle.load(f)

            with open(f'{self.data_original_path}val_data.pkl', 'rb') as f:
                val_data = pickle.load(f)

            with open(f'{self.data_original_path}test_data.pkl', 'rb') as f:
                test_data = pickle.load(f)

            # Preprocess data with chosen imputation method
            train_data, val_data, test_data = preprocess_general(train_data, val_data, test_data, imputation_method)

        elif self.data_source == 'MIMIC-III-Challenge':
            train_data, val_data, test_data = DataChallenge(imputation_method=imputation_method).get_data()


        data_provider = Data(train_data, val_data, test_data, norm_method)
        return data_provider

    def _train_predict_dl(self, data_provider, model_name, params):
        x_train_norm, y_train_norm = data_provider.get_train_data_dl()
        x_val_norm, y_val_norm = data_provider.get_val_data_dl()
        x_test_norm = data_provider.get_test_data_dl()

        model = self.models[model_name](model_name, params)

        model.fit(x_train_norm, y_train_norm, x_val_norm, y_val_norm)
        return model.predict(x_test_norm), model.get_train_time()

    def _train_predict_ml(self, data_provider, model_name, params):

        x_train_norm, y_train_norm = data_provider.get_train_data_ml()
        x_test_norm = data_provider.get_test_data_ml()

        model = self.models[model_name](params)

        model.fit(x_train_norm, y_train_norm)
        return model.predict(x_test_norm), model.get_train_time()

    def _train_predict(self, type, *args):
        if type == 'dl':
            prediction, train_time = self._train_predict_dl(*args)
        else:
            prediction, train_time = self._train_predict_ml(*args)
        return prediction, train_time

    def run(self):
        columns = ['model', 'imputation_method', 'norm_method', 'params', 'model_id', 'time', *self.metric_calculator.get_name_metrics()]
        results = pd.DataFrame(columns=columns)
        save_path = f'./results/{self.data_source}/{self.name}/optimization'

        os.makedirs(save_path, exist_ok=True)

        for imputation_method, norm_method in itertools.product(self.imputation_methods, ['minmax']):
            data_provider = self._load_data(imputation_method, norm_method)

            for model_name in self.models_to_create.keys():

                if self.models_to_create[model_name] == 1:

                    model_id = 0
                    results_path = f'{save_path}/predictions/{imputation_method}/{norm_method}/{model_name}/'

                    os.makedirs(results_path, exist_ok=True)

                    model_type = 'dl' if model_name in self.dl_models else 'ml'

                    for params in ParamsGrid(model_name, model_type, self.iter_sampler):

                        prediction, training_time = self._train_predict(model_type, data_provider, model_name, params)
                            
                        real = data_provider.get_y_test_real()
                        np.save(f'{results_path}{model_id}.npy', prediction)

                        metrics = self.metric_calculator.get_metrics(real, prediction)

                        row = [model_name, imputation_method, norm_method, params, model_id, training_time, *metrics]
                        results.loc[results.shape[0]] = row

                        results.to_csv(f'{save_path}/results.csv')
                        model_id += 1
        