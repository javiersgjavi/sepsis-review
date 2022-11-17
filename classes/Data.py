from inspect import Parameter
import json
import itertools
import numpy as np

from sklearn.model_selection import ParameterSampler

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
                res[i, j, :] = (data[i, j, :] - self.param_1[j])/(self.param_2[j] - self.param_1[j])
            else:
                res[i, j, :] = (data[i, j, :] - self.param_1[j])/self.param_2[j]
        return res

class Data:
    def __init__(self, train_data, val_data, test_data, norm_method='minmax'):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        print('Calculating normalization params for DL datasets...')
        self.norm_dl = Normalizer_DL(np.concatenate([self.train_data[0], self.val_data[0], self.test_data[0]], axis=0), method=norm_method)
        
        print('Calculating normalization params for ML datasets...')
        self.norm_ml = Normalizer(np.concatenate([self.train_data[1], self.val_data[1], self.test_data[1]], axis=0), method=norm_method)
        

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
