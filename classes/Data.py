from inspect import Parameter
import os
import pickle
import pandas as pd
import json
import itertools
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import ParameterSampler
from utils.preprocess_data import linear_interpolation, carry_forward_data, forward_filling, indicator_imputation


class ParamsGrid:
    def __init__(self, model_name, model_type, iter_sampler):
        self.i = 0
        self.model_type = model_type
        self.model_name = model_name
        self.sampler = self.load_params(iter_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.sampler)

    def load_params(self, iter_sampler):
        params_dict = {}

        with open(f'./parameters.json', 'r') as file:
            parameters = json.load(file)

        if self.model_type == 'dl':
            for param in ['batch_size', 'learning_rate']:
                params_dict[param] = parameters[param]

        for param in parameters['model_params'][self.model_name].keys():
            params_dict[param] = parameters['model_params'][self.model_name][param]

        return iter(ParameterSampler(params_dict, n_iter=iter_sampler))


class Normalizer:
    def __init__(self, data, method='minmax'):
        self.method = method
        self.param_1, self.param_2 = self.get_norm_params(data)

    def get_norm_params(self, data):
        if self.method == 'minmax':
            param_1 = np.min(data, axis=0)
            param_2 = np.max(data, axis=0)
            param_2 = np.where(param_2 == 0, 0.01, param_2)

        elif self.method == 'zscore':
            param_1 = np.mean(data, axis=0)
            param_2 = np.std(data, axis=0)

            param_2 = np.where(param_2 == 0, 0.01, param_2)

        return param_1, param_2

    def normalize(self, data):
        if self.method == 'minmax':
            res = (data - self.param_1) / (self.param_2 - self.param_1)

        elif self.method == 'zscore':
            res = (data - self.param_1) / self.param_2

        return res

    def denormalize(self, data):
        if self.method == 'minmax':
            res = data * (self.param_2 - self.param_1) + self.param_1

        elif self.method == 'zscore':
            res = data * self.param_2 + self.param_1

        return res


class Normalizer_DL:
    def __init__(self, data, method='minmax'):
        self.method = method
        self.param_1, self.param_2 = self.min_max_params(data) if method == 'minmax' else self.zscore_params(data)

    def min_max_params(self, data):
        param_1 = np.zeros(data.shape[1])
        param_2 = np.zeros(data.shape[1])

        for j in range(data.shape[1]):
            param_1[j] = np.min(data[:, j, :])
            param_2[j] = np.max(data[:, j, :])

        return param_1, param_2

    def zscore_params(self, data):

        param_1 = np.zeros(data.shape[1])
        param_2 = np.zeros(data.shape[1])

        for j in range(data.shape[1]):
            param_1[j] = np.mean(data[:, j, :])
            param_2[j] = np.std(data[:, j, :])

        return param_1, param_2

    def normalize(self, data):
        res = np.zeros(data.shape)

        for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
            if self.method == 'minmax':
                res[i, j, :] = (data[i, j, :] - self.param_1[j]) / (self.param_2[j] - self.param_1[j])
            else:
                res[i, j, :] = (data[i, j, :] - self.param_1[j]) / self.param_2[j]
        return res


class Data:
    def __init__(self, train_data, val_data, test_data, norm_method='minmax'):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        print('Calculating normalization params for DL datasets...')
        self.norm_dl = Normalizer_DL(np.concatenate([self.train_data[0], self.val_data[0], self.test_data[0]], axis=0),
                                     method=norm_method)

        print('Calculating normalization params for ML datasets...')
        self.norm_ml = Normalizer(np.concatenate([self.train_data[1], self.val_data[1], self.test_data[1]], axis=0),
                                  method=norm_method)

    def get_train_data_dl(self, normalize=True):
        x = self.train_data[0]
        y = self.train_data[2]

        if normalize:
            x = self.norm_dl.normalize(x)

        return x, y

    def get_val_data_dl(self, normalize=True):
        x = self.val_data[0]
        y = self.val_data[2]

        if normalize:
            x = self.norm_dl.normalize(x)

        return x, y

    def get_test_data_dl(self, normalize=True):
        x = self.test_data[0]

        if normalize:
            x = self.norm_dl.normalize(x)

        return x

    def get_train_data_ml(self, normalize=True):
        x = np.concatenate([self.train_data[1], self.val_data[1]], axis=0)
        y = np.concatenate([self.train_data[2], self.val_data[2]], axis=0)

        if normalize:
            x = self.norm_ml.normalize(x)

        return x, y

    def get_test_data_ml(self, normalize=True):
        x = self.test_data[1]

        if normalize:
            x = self.norm_ml.normalize(x)

        return x

    def get_y_test_real(self):
        return self.test_data[2]


class DataChallenge:
    def __init__(self, imputation_method, val_ratio=0.1, test_ratio=0.1, path='./data/challenge/original/Dataset.csv'):
        self.imputation_method = imputation_method
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.path = path

    def load_csv(self):
        data = pd.read_csv(self.path)
        data = data.drop(columns=['Unnamed: 0'])
        patient_ids = data['Patient_ID'].unique()
        print(f'Num of patients: {len(patient_ids)}')

        return data, patient_ids

    def get_dict_patients_data(self, data, patient_ids):
        patients_data = {}
        for patient_id in tqdm(patient_ids, desc='Get patients data'):

            patient_data = data[data['Patient_ID'] == patient_id]
            patient_data = patient_data.drop(columns=['Patient_ID'])

            if patient_data['SepsisLabel'].sum() > 0:
                onset_row = patient_data[patient_data['SepsisLabel'] == 1].first_valid_index()
                patient_data = patient_data.loc[:onset_row]

            patients_data[patient_id] = patient_data

        return patients_data

    def pad_to_48h(self, patient):
        patient = patient.reset_index(drop=True)
        patient = patient.drop(columns=['Hour'])

        pad_size = 48 - patient.shape[0]
        patient.index = patient.index + pad_size
        patient = patient.reindex(np.arange(48))

        return patient

    def get_48h_window(self, patient):
        patient_window = patient.iloc[-48:]
        patient_window = patient_window.reset_index(drop=True)
        patient_window = patient_window.drop(columns=['Hour'])

        return patient_window

    def get_transform_data(self, patient_dict):
        for patient_id in tqdm(patient_dict.keys(), desc='Transform data'):
            patient = patient_dict[patient_id]
            if patient.shape[0] < 48:
                patient_dict[patient_id] = self.pad_to_48h(patient)
            else:
                patient_dict[patient_id] = self.get_48h_window(patient)

        data_list = list(patient_dict.values())

        data_array = np.stack(data_list)

        return data_array.transpose(0, 2, 1)

    def get_x_y_data(self, patients_array):
        x_data = patients_array[:, :-1, :]
        x_dl = np.nan_to_num(x_data, 0)
        x_ml = np.zeros((x_dl.shape[0], x_dl.shape[1] * x_dl.shape[2]))

        y_data = patients_array[:, -1, :].max(axis=1)
        y_data = np.nan_to_num(y_data, 0)

        for i in tqdm(range(x_data.shape[0]), desc='Imputation'):
            row = x_dl[i]
            if self.imputation_method == 'linear_interpolation':
                row_preprocessed = linear_interpolation(row)

            elif self.imputation_method == 'carry_forward':
                row_preprocessed = carry_forward_data(row)

            elif self.imputation_method == 'forward_filling':
                row_preprocessed = forward_filling(row)

            elif self.imputation_method == 'zero_imputation':
                row_preprocessed = row

            elif self.imputation_method == 'indicator_imputation':
                row_preprocessed = indicator_imputation(row)

            x_dl[i] = row_preprocessed
            x_ml[i] = row_preprocessed.flatten()

        return [x_dl, x_ml, y_data]

    def split_data(self, data):
        x_dl, x_ml, y_data = data
        pos_val = int(x_dl.shape[0] * self.val_ratio)
        pos_test = int(x_dl.shape[0] * self.test_ratio)

        x_dl_train = x_dl[:-pos_val - pos_test]
        x_ml_train = x_ml[:-pos_val - pos_test]
        y_train = y_data[:-pos_val - pos_test]

        x_dl_val = x_dl[-pos_val - pos_test:-pos_test]
        x_ml_val = x_ml[-pos_val - pos_test:-pos_test]
        y_val = y_data[-pos_val - pos_test:-pos_test]

        x_dl_test = x_dl[-pos_test:]
        x_ml_test = x_ml[-pos_test:]
        y_test = y_data[-pos_test:]

        return [x_dl_train, x_ml_train, y_train], [x_dl_val, x_ml_val, y_val], [x_dl_test, x_ml_test, y_test]

    def generate_data(self, folder_data):
        data_csv, patient_ids = self.load_csv()
        patients_dict_data = self.get_dict_patients_data(data_csv, patient_ids)
        patients_array = self.get_transform_data(patients_dict_data)
        data = self.get_x_y_data(patients_array)
        train, val, test = self.split_data(data)

        with open(f'{folder_data}/train.pkl', 'wb') as f:
            pickle.dump(train, f)

        with open(f'{folder_data}/val.pkl', 'wb') as f:
            pickle.dump(val, f)

        with open(f'{folder_data}/test.pkl', 'wb') as f:
            pickle.dump(test, f)

        return train, val, test

    def load_final_data(self, path):
        with open(f'{path}/train.pkl', 'rb') as f:
            train = pickle.load(f)

        with open(f'{path}/val.pkl', 'rb') as f:
            val = pickle.load(f)

        with open(f'{path}/test.pkl', 'rb') as f:
            test = pickle.load(f)

        return train, val, test

    def get_data(self):
        folder_data = f'./data/challenge/{self.imputation_method}'
        os.makedirs(folder_data, exist_ok=True)

        if len(os.listdir(folder_data)) == 0:
            self.generate_data(folder_data)

        train, val, test = self.load_final_data(folder_data)

        return train, val, test
