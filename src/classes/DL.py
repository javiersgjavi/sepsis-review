from gc import callbacks
import time
import numpy as np
import tensorflow as tf
from tcn import TCN as TCN_layer

class DL:
    def __init__(self, name, params):
        self.model = None
        self.name = name
        self.params = params

    def fit(self, x_train, y_train, x_val, y_val):

        self.build_model(x_train.shape)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        t = time.time()
        
        self.model.fit(x_train, y_train, epochs=100, batch_size=self.params['batch_size'], validation_data=(x_val, y_val), callbacks=[early_stopping])
        self.train_time = (time.time() - t)/60
        print(f'{self.name} trained in {round(self.train_time, 2)} minutes')

    def predict(self, x_test):
        prediction = self.model.predict(x_test)
        return prediction 

    def get_train_time(self):
        return self.train_time

class LSTM(DL):

    def build_model(self, shape_x):

        last_activation_function = 'sigmoid'
        loss = 'binary_crossentropy'

        rows = shape_x[1]
        columns = shape_x[2]

        recurrent_units = [self.params['units']] * self.params['layers']
        return_sequence = self.params['return_sequence']
        recurrent_dropout = self.params['recurrent_dropout']

        # First layer
        inputs = tf.keras.layers.Input(shape=(rows, columns))

        # First LSTM layer
        return_sequence_tmp = return_sequence if len(recurrent_units) == 1 else True
        x = tf.keras.layers.LSTM(recurrent_units[0], return_sequences=return_sequence_tmp, dropout=recurrent_dropout, activation='tanh')(inputs)

        # Hidden LSTM layers
        for i, units in enumerate(recurrent_units[1:]):
            return_sequence_tmp = return_sequence if i == len(recurrent_units) - 2 else True
            x = tf.keras.layers.LSTM(units, return_sequences=return_sequence_tmp, dropout=recurrent_dropout, activation='tanh')(x)

        # Hidden Dense layers
        if return_sequence:
            x = tf.keras.layers.Flatten()(x)

        for hidden_units in self.params['dense_layers']:
            x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
            if self.params['dense_dropout'] > 0:
                x = tf.keras.layers.Dropout(self.params['dense_dropout'])(x)
                
        # Output layer
        x = tf.keras.layers.Dense(1, activation=last_activation_function)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']), loss=loss, metrics=['accuracy'])
        
        self.model = model

class GRU(DL):

    def build_model(self, shape_x):

        last_activation_function = 'sigmoid'
        loss = 'binary_crossentropy'
        
        rows = shape_x[1]
        columns = shape_x[2]

        recurrent_units = [self.params['units']] * self.params['layers']
        return_sequence = self.params['return_sequence']
        recurrent_dropout = self.params['recurrent_dropout']

        # First layer
        inputs = tf.keras.layers.Input(shape=(rows, columns))

        # First GRU layer
        return_sequence_tmp = return_sequence if len(recurrent_units) == 1 else True
        x = tf.keras.layers.GRU(recurrent_units[0], return_sequences=return_sequence_tmp, dropout=recurrent_dropout, activation='tanh')(inputs)

        # Hidden GRU layers
        for i, units in enumerate(recurrent_units[1:]):
            return_sequence_tmp = return_sequence if i == len(recurrent_units) - 2 else True
            x = tf.keras.layers.GRU(units, return_sequences=return_sequence_tmp, dropout=recurrent_dropout, activation='tanh')(x)


        # Hidden Dense layers
        if return_sequence:
            x = tf.keras.layers.Flatten()(x)

        for hidden_units in self.params['dense_layers']:
            x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
            if self.params['dense_dropout'] > 0:
                x = tf.keras.layers.Dropout(self.params['dense_dropout'])(x)
                
        # Output layer
        x = tf.keras.layers.Dense(1, activation=last_activation_function)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']), loss=loss)
        
        self.model = model

class TCN(DL):

    def build_model(self, shape_x):

        last_activation_function = 'sigmoid'
        loss = 'binary_crossentropy'
        
        rows = shape_x[1]
        columns = shape_x[2]

        inputs = tf.keras.layers.Input(shape=(rows, columns))
        x = TCN_layer(
            nb_filters=self.params['nb_filters'],
            kernel_size=self.params['kernel_size'],
            nb_stacks=self.params['nb_stacks'],
            dilations=self.params['dilations'],
            use_skip_connections=True,
            dropout_rate=self.params['tcn_dropout'],
            activation='relu',
            use_batch_norm=False,
            padding='causal',
            return_sequences=self.params['return_sequences'])(inputs)

        if self.params['return_sequences']:
            x = tf.keras.layers.Flatten()(x)

         # Hidden Dense layers
        for hidden_units in self.params['dense_layers']:
            x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
            if self.params['dense_dropout'] > 0:
                x = tf.keras.layers.Dropout(self.params['dense_dropout'])(x)
                
        # Output layer
        x = tf.keras.layers.Dense(1, activation=last_activation_function)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']), loss=loss)
        
        self.model = model

class CNN(DL):

    def build_model(self, shape_x):

        last_activation_function = 'sigmoid'
        loss = 'binary_crossentropy'
        
        rows = shape_x[1]
        columns = shape_x[2]

        conv_layers = [i[0] for i in self.params['conv_blocks']]
        kernel_sizes = [i[1] for i in self.params['conv_blocks']]
        pool_sizes = [i[2] for i in self.params['conv_blocks']]

        # Input layer
        inputs = tf.keras.layers.Input(shape=(rows, columns))

        # First Convolutional layer
        x = tf.keras.layers.Conv1D(conv_layers[0], kernel_sizes[0], activation='relu', padding='same')(inputs)
        if pool_sizes[0] and x.shape[-2] // pool_sizes[0] > 1:
            x = tf.keras.layers.MaxPooling1D(pool_sizes[0])(x)

        # Hidden Convolutional layers
        for chanels, kernel_size, pool_size in zip(conv_layers[1:], kernel_sizes[1:], pool_sizes[1:]):
            x = tf.keras.layers.Conv1D(chanels, kernel_size, activation='relu', padding='same')(x)
            if pool_size and x.shape[-2] // pool_size > 1:
                x = tf.keras.layers.MaxPooling1D(pool_size)(x)

        # Hidden Dense layers
        x = tf.keras.layers.Flatten()(x)
        for hidden_units in self.params['dense_layers']:
            x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)
            if self.params['dense_dropout'] > 0:
                x = tf.keras.layers.Dropout(self.params['dense_dropout'])(x)
                
        # Output layer
        x = tf.keras.layers.Dense(1, activation=last_activation_function)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']), loss=loss)
        
        self.model = model

class MLP(DL):

    def build_model(self, shape_x):

        last_activation_function = 'sigmoid'
        loss = 'binary_crossentropy'
        
        rows = shape_x[1]
        columns = shape_x[2]

        # Input layer
        inputs = tf.keras.layers.Input(shape=(rows, columns))

        # Hidden layers
        x = tf.keras.layers.Flatten()(inputs)

        for hidden_units in self.params['hidden_layers']:
            x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)

        # Output layer
        x = tf.keras.layers.Dense(1, activation=last_activation_function)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']), loss=loss)
        
        self.model = model