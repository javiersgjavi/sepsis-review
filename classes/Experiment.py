import pickle
import sys
import inspect
import pandas as pd
import os
import numpy as np

from classes.Metrics import MetricCalculator
from classes.Data import Data, DataChallenge
from classes.DL import GRU, TCN, CNN, LSTM, MLP
from classes.ML import LinearSVC, XGBClassifier, LogisticRegression, AdaBoostClassifier, RandomForestClassifier
from utils.preprocess_data import process_params_string

class Experiment:
    def __init__(self, data, name, hours_before_onset):

        self.data_source = data

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

        self.base_path = f'./results/{data}/{name}/experiment/'
        self.results_optimization = pd.read_csv(f'./results/{data}/{name}/optimization/results.csv', index_col='Unnamed: 0')
        self.hours_before_onset = hours_before_onset
        self.dl_models = [cls_name for cls_name, cls_obj in inspect.getmembers(sys.modules['classes.DL']) if inspect.isclass(cls_obj)]
        self.data_generated = {}

    def get_params_model(self, model):
        best_model = self.results_optimization[self.results_optimization['model'] == model]['auprc'].idxmax()
        row_best_model = self.results_optimization.iloc[best_model]
        imputation_method = row_best_model['imputation_method']
        norm_method = row_best_model['norm_method']
        params = process_params_string(row_best_model['params'])
        auprc = row_best_model['auprc']
        roc_auc = row_best_model['roc_auc']
        accuracy = row_best_model['accuracy']

        return imputation_method, norm_method, params, auprc, roc_auc, accuracy

    def remove_hours(self, train_data, val_data, test_data, i):
        measures = train_data[0].shape[1]
        times = train_data[0].shape[2]

        # Removes hours of the dl dataset
        train_data[0], val_data[0], test_data[0] = train_data[0][:, :, :-i], val_data[0][:, :, :-i], test_data[0][:, :, :-i]

        for dataset in [train_data, val_data, test_data]:

            # Removes de hours of the ml dataset
            for measure in range(1, measures+1):
                mask = np.ones(dataset[1].shape[1], dtype=bool)
                start_index = (measure*times) - i
                end_index = (measure*times)
                mask[start_index:end_index] = False
            dataset[1] = dataset[1][:, mask]
        
        print(train_data[0].shape)
        return train_data, val_data, test_data

    def load_data(self, imputation_method, norm_method, i):
        name_data = f'{self.data_source}_{imputation_method}_{norm_method}_{i}'

        if name_data in self.data_generated.keys():
            data_provider = self.data_generated[name_data]

        else:
            if self.data_source == 'MIMIC-III':
                path_data = f'./data/{imputation_method}/'
                with open(f'{path_data}train_data', 'rb') as f:
                    train_data = pickle.load(f)

                with open(f'{path_data}val_data', 'rb') as f:
                    val_data = pickle.load(f)

                with open(f'{path_data}test_data', 'rb') as f:
                    test_data = pickle.load(f)

            elif self.data_source == 'MIMIC-III-Challenge':
                train_data, val_data, test_data = DataChallenge(imputation_method=imputation_method).get_data()

            train_data, val_data, test_data = self.remove_hours(train_data, val_data, test_data, i)
            data_provider = Data(train_data, val_data, test_data, norm_method)
            self.data_generated[name_data] = data_provider

        return data_provider

    def train_predict_dl(self, model_name, params, data_provider):
        x_train_norm, y_train_norm = data_provider.get_train_data_dl()
        x_val_norm, y_val_norm = data_provider.get_val_data_dl()
        x_test_norm = data_provider.get_test_data_dl()

        model = self.models[model_name](model_name, params)

        model.fit(x_train_norm, y_train_norm, x_val_norm, y_val_norm)
        return model.predict(x_test_norm)

    def train_predict_ml(self, model_name, params, data_provider):
        x_train_norm, y_train_norm = data_provider.get_train_data_ml()
        x_test_norm = data_provider.get_test_data_ml()

        print(f'Shape of X: {x_train_norm.shape}')
        model = self.models[model_name](params)

        model.fit(x_train_norm, y_train_norm)
        return model.predict(x_test_norm)

    def train_predict(self, model_name, params, data_provider):
        if model_name in self.dl_models:
            prediction = self.train_predict_dl(model_name, params, data_provider)

        else:
            prediction = self.train_predict_ml(model_name, params, data_provider)

        return prediction
        
    def run(self):

        column_names = ['t'] + [f't-{i+1}' for i in range(self.hours_before_onset)]

        results_auprc = pd.DataFrame(columns = column_names)
        results_roc_auc = pd.DataFrame(columns = column_names)
        results_accuracy = pd.DataFrame(columns = column_names)

        for model_name in self.results_optimization['model'].unique():

            path_model = f'{self.base_path}/predictions/{model_name}/'
            os.makedirs(path_model, exist_ok=True)
            imputation_method, norm_method, params, auprc, roc_auc, accuracy = self.get_params_model(model_name)
            
            row_auprc = [auprc]
            row_au_roc = [roc_auc]
            row_accuracy = [accuracy]

            for i in range(1, self.hours_before_onset+1):

                data_provider = self.load_data(imputation_method, norm_method, i)

                prediction = self.train_predict(model_name, params, data_provider)
                np.save(f'{path_model}t-{i}.npy', prediction)

                real = data_provider.get_y_test_real()
                metrics = self.metric_calculator.get_metrics(real, prediction)

                row_auprc.append(metrics[0])
                row_au_roc.append(metrics[1])
                row_accuracy.append(metrics[2])

            results_auprc.loc[model_name, :] = row_auprc
            results_roc_auc.loc[model_name, :] = row_au_roc
            results_accuracy.loc[model_name, :] = row_accuracy

        results_auprc.to_csv(f'{self.base_path}/auprc.csv')
        results_roc_auc.to_csv(f'{self.base_path}/roc_auc.csv')
        results_accuracy.to_csv(f'{self.base_path}/accuracy.csv')
