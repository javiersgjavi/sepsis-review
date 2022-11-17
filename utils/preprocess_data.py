import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from classes.MGP import *
import gpytorch, torch

def get_max_size(data):
    max_size = 0
    for row in data:
        if row.shape[0] > max_size:
            max_size = row.shape[0]
    return max_size


def fill_padded_array(res, data):
    for i in range(res.shape[0]):
        res[i, :data[i].shape[0]] = data[i]
    return res

def pad_data(train, val, test):
    train_max = get_max_size(train)
    val_max = get_max_size(val)
    test_max = get_max_size(test)

    max_size = max(train_max, val_max, test_max)

    train_padded = np.zeros((train.shape[0], max_size))
    val_padded = np.zeros((val.shape[0], max_size))
    test_padded = np.zeros((test.shape[0], max_size))

    train_padded = fill_padded_array(train_padded, train)
    val_padded = fill_padded_array(val_padded, val)
    test_padded = fill_padded_array(test_padded, test)

    return np.concatenate((train_padded, val_padded, test_padded), axis=0).astype(int)

# get number of different measures of the original data and number of bins for the new dataset
def get_bins_and_measures(train, val, test):
    train = np.array(train)
    val = np.array(val)
    test = np.array(test)

    train_times, val_times, test_times = train[1], val[1], test[1]
    train_measures, val_measures, test_measures = train[2], val[2], test[2]
    
    # transform all data to the sime size of the longest one
    times = pad_data(train_times, val_times, test_times)
    measures = pad_data(train_measures, val_measures, test_measures)

    bins = np.sort(np.unique(times))
    measures_uniques = np.sort(np.unique(measures))

    return bins, measures_uniques

def carry_forward_data(row):
    for i in range(row.shape[0]):
        mean = np.mean(row[i])
        last_value = 0
        for j in range(row.shape[1]):
            if row[i][j] == 0:
                if last_value == 0:
                    row[i][j] = mean
                else:
                    row[i][j] = last_value          
            else:
                last_value = row[i][j]
    return row

def linear_interpolation(array):
    '''Calculated the linear interpolation of a vector'''

    for measure in range(array.shape[0]):

        i = 0
        if np.any(array[measure,:] !=  0):
            while i < array.shape[1]:

                if array[measure, i] == 0:
                
                    base_value = array[measure, i] if i == 0 else array[measure, i-1]

                    # Find the next non-zero value
                    for next_value in range(i+1, array.shape[1]):
                        if array[measure, next_value] != 0:
                            break

                    # Calculate the increment
                    steps = next_value - i
                    increment = (array[measure, next_value]- base_value)/(steps+1)

                    # Fill the vector
                    for j in range(steps):
                        interpolated_value = base_value + (j+1)*increment
                        array[measure, i+j] = float(interpolated_value)
                    
                    i = next_value

                else:
                    i += 1

                # If there are no more non-zero values, break
                if not np.any(array[measure, i:] != 0):
                    break

    return array

def forward_filling(array):

    for measure in range(array.shape[0]):

        last_value = 0
        for time in range(array.shape[1]):
            if array[measure, time] == 0:
                array[measure, time] = last_value
                
            else:
                last_value = array[measure, time]

    return array

def indicator_imputation(array):
    new_array = np.zeros((array.shape[0]*2, array.shape[1]))
    new_array[:array.shape[0],:] = array

    for measure in range(array.shape[0]):
        for column in range(array.shape[1]):
            if array[measure, column] == 0:
                new_array[measure+array.shape[0], column] = 1

    return array

def preprocess(data, bins, measures_values, imputation_method, set):
    x_dl = []
    x_ml = []

    y = data[4] # get the labels

    values_dim = data[0]
    times_dim = data[1]
    measures_dim = data[2]
    index_time_dim = data[3]
    

    # Transform data to array -> (patient, measure, time)
    for i in tqdm(range(values_dim.shape[0]), desc=f'Preprocessing {set} set'):
        row = np.zeros((measures_values.shape[0], bins.shape[0]))
        bins_to_mean = {}
        for j in range(values_dim[i].shape[0]):
            value= values_dim[i][j]
            index_time = index_time_dim[i][j]

            measure = int(measures_dim[i][j])
            time = int(times_dim[i][index_time])

            if row[measure, time] == 0:
                row[measure, time] = value

            elif (measure, time) in bins_to_mean.keys():

                bins_to_mean[(measure, time)].append(value)
            else:
                bins_to_mean[(measure, time)] = [value]

        for key in bins_to_mean.keys():
            measure, time = key
            list_values = bins_to_mean[key]
            list_values.append(row[measure, time])
            row[measure, time] = np.mean(list_values)

        # Apply the imputation method to fixed arrays
        if imputation_method == 'linear_interpolation':
            row_preprocessed = linear_interpolation(row)

        elif imputation_method == 'carry_forward':
            row_preprocessed = carry_forward_data(row)

        elif imputation_method == 'forward_filling':
            row_preprocessed = forward_filling(row)

        elif imputation_method == 'zero_imputation':
            row_preprocessed = row
            
        elif imputation_method == 'indicator_imputation':
            row_preprocessed = indicator_imputation(row)
        
        x_dl.append(row_preprocessed)
        x_ml.append(row_preprocessed.flatten())

    return [np.array(x_dl), np.array(x_ml), y]

# Method that preprocesses the data except for mgp imputation
def preprocess_general(train_data, val_data, test_data, imputation_method):

    path_data = f'./data/{imputation_method}/'
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    if len(os.listdir(path_data)) == 3:
        with open(f'{path_data}train_data', 'rb') as f:
            train_data = pickle.load(f)

        with open(f'{path_data}val_data', 'rb') as f:
            val_data = pickle.load(f)

        with open(f'{path_data}test_data', 'rb') as f:
            test_data = pickle.load(f)

    else:
        train_data, val_data, test_data = slide_time_window([train_data, val_data, test_data])
        bins, measures = get_bins_and_measures(train_data, val_data, test_data)
        train_data = preprocess(train_data, bins, measures, imputation_method, 'training')
        val_data = preprocess(val_data , bins, measures, imputation_method, 'validation')
        test_data  = preprocess(test_data, bins, measures, imputation_method, 'test')

        with open(f'{path_data}train_data', 'wb') as f:
            pickle.dump(train_data, f)

        with open(f'{path_data}val_data', 'wb') as f:
            pickle.dump(val_data, f)

        with open(f'{path_data}test_data', 'wb') as f:
            pickle.dump(test_data, f)

    return train_data, val_data, test_data
    
def train_gp_model(i, j, times, values_dim, measures_dim, bins, measures):
    measures_uniques = []

    indices_time_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in times[i:j]], batch_first=True)
    values_dim_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in values_dim[i:j]], batch_first=True)
    measures_dim_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in measures_dim[i:j]], batch_first=True)

    measures_uniques = np.sort(measures)
    
    indices_time_batch = torch.reshape(indices_time_batch, (indices_time_batch.shape[0], indices_time_batch.shape[1], 1))
    measures_dim_batch = torch.reshape(measures_dim_batch, (measures_dim_batch.shape[0], measures_dim_batch.shape[1], 1))

    measures_uniques = np.sort(list(measures_uniques))

    test_indices = torch.from_numpy(np.array([np.repeat(np.array(measures_uniques), len(bins)) for i in range(j-i)]))
    test_indices = torch.reshape(test_indices, (test_indices.shape[0], test_indices.shape[1], 1))

    test_inputs = torch.from_numpy(np.array([np.repeat(np.array(bins), len(measures_uniques)) for i in range(j-i)]))
    test_inputs = torch.reshape(test_inputs, (test_inputs.shape[0], test_inputs.shape[1], 1))

    valid_lengths = torch.Tensor(j-i)
    for i in range(indices_time_batch.shape[0]):
        values = np.unique(indices_time_batch[i,:,0])
        valid_lengths[i] = np.ceil(np.max(values))

    data = indices_time_batch.float(), measures_dim_batch.float(), values_dim_batch.float(), test_indices.float(), test_inputs.float(), valid_lengths.float()
    gp = GPAdapter(
        train_x=(data[0], data[1]),
        train_y=data[3],
        n_mc_smps=10,
        sampling_type='monte_carlo',
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        output_device='cpu',
        num_tasks=len(bins),
        n_devices=1,
        kernel='rbf',
        mode='normal',
        keops=False
    )

    return gp.forward(*data)

def get_gp_predictions(data, bins, measures, set):
    x_dl = []
    x_ml = []

    y = data[4] # is case or control

    values_dim = data[0]#.astype('float64')
    times_dim = data[1]#.astype('int')
    measures_dim = data[2]#.astype('int')
    index_time_dim = data[3]


    times = []

    for matrix in range(len(index_time_dim)):
        row = []
        for value in range(len(index_time_dim[matrix])):
            index_of_time = index_time_dim[matrix][value]
            value_time = times_dim[matrix][index_of_time]
            row.append(value_time)
        times.append(row)

    batch = 64
    predictions = []

    for i in tqdm(range(0, len(values_dim)-batch, batch), desc=f'Preprocessing {set} set'):

        x = train_gp_model(i, i+batch, times, values_dim, measures_dim, bins, measures)
        predictions.append(x)

    if i + batch < len(values_dim) - 1:
        x = train_gp_model(i+batch, len(values_dim), times, values_dim, measures_dim, bins, measures)
        predictions.append(x)

    for i in range(len(predictions)):
        batch = predictions[i]
        for j in range(batch.shape[0]):
            row = batch[j].numpy()
            x_dl.append(row)
            x_ml.append(row.flatten())

    x_ml = np.array(x_ml)
    x_ml_formated = np.zeros((x_ml.shape[0], x_ml[0].shape[0]))
    x_ml_formated[:] = np.array(x_ml)
    return [np.array(x_dl), x_ml_formated, y]

def preprocess_gp(train_data, val_data, test_data):

    path_start = f'./data/gaussian_process/'

    if not os.path.exists(path_start):
        os.makedirs(path_start)

    if not os.path.exists(f'{path_start}train_data.pkl'):
        print(f'preprocessed data with MGP does not exist, starting generation...')
        train_data, val_data, test_data = slide_time_window([train_data, val_data, test_data])
        bins, measures = get_bins_and_measures(train_data, val_data, test_data)

        train_data = get_gp_predictions(train_data, bins, measures, 'training')
        with open(f'{path_start}train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)

        val_data = get_gp_predictions(val_data, bins, measures, 'validation')
        with open(f'{path_start}val_data.pkl', 'wb') as f:
            pickle.dump(val_data, f)

        test_data = get_gp_predictions(test_data, bins, measures, 'test')
        with open(f'{path_start}test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)

    else:

        with open(f'{path_start}train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)

        with open(f'{path_start}val_data.pkl', 'rb') as f:
            val_data = pickle.load(f)

        with open(f'{path_start}test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)

        print('Loaded preprocessed data with MGP')

    
    return train_data, val_data, test_data
    
def slide_time_window(data):
    res = []
    for dataset in data:
        for i, times in enumerate(dataset[1]):
            max_time = np.max(times)
            if max_time < 48:
                desp = 48 - max_time
            elif max_time > 48:
                desp = -(max_time - 48)

            times += desp
    
            dataset[1][i] = times

        res.append(dataset)

    return res

def process_params_string(params):
    params = params.replace("'", '"')
    params = params.replace('False', 'false')
    params = params.replace('True', 'true')

    return json.loads(params)
