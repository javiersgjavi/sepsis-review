import os
import json
import pickle
import numpy as np
from tqdm import tqdm

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
