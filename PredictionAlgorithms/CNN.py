#CNN

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import random
import tensorflow as tf

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Load data from multiple sheets
excel_file = r"Cucumber_FillKNN.xlsx"
sheet_names = pd.ExcelFile(excel_file).sheet_names

# Values of dayPredict for each code
dayPredict_values = [90]

# Output Excel file
output_file = "AllData/Cucumber_CNN_7seq_90day.xlsx"

# Create an Excel writer object to save the results
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Loop through each sheet
    for sheet_name in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        # Extract 'WholesalePriceNew' column
        data = df['WholesalePriceNew']
        timesteps = 7
        epochs = 100

        # Loop through each value of dayPredict
        for dayPredict in dayPredict_values:
            # Convert to np array
            dataset = data.values.reshape(-1, 1)  # Reshape to 2D array

            # Data preprocessing
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            trainLength = math.ceil(len(dataset) - dayPredict)

            # Training data creation
            train_data = scaled_data[:trainLength]
            x_train, y_train = [], []
            for i in range(timesteps, len(train_data)):
                x_train.append(train_data[i - timesteps:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Model building
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, 1)))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(x_train, y_train, batch_size=32, epochs=epochs)

            # Test data preparation
            test_data = scaled_data[trainLength - timesteps:, :]
            x_test = []
            y_test = dataset[trainLength:]
            for i in range(timesteps, len(test_data)):
                x_test.append(test_data[i - timesteps:i, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Model evaluation
            predictionsCNN = model.predict(x_test)
            predictionsCNN = scaler.inverse_transform(predictionsCNN)
            rmse_CNN = sqrt(mean_squared_error(predictionsCNN, y_test))
            mae_CNN = mean_absolute_error(predictionsCNN, y_test)
            mse_CNN = mean_squared_error(predictionsCNN, y_test)
            mape_CNN = mean_absolute_percentage_error(predictionsCNN, y_test) * 100

            print(f'\nResults for sheet {sheet_name} with dayPredict={dayPredict}:')
            print('RMSE for CNN Model is:', rmse_CNN)
            print('MAE for CNN Model is:', mae_CNN)
            print('MSE for CNN Model is:', mse_CNN)
            print('MAPE for CNN Model is:', mape_CNN)

            # Save predictions to file
            df_output = pd.DataFrame({
                'Actual': y_test.flatten(),
                'Predictions': predictionsCNN.flatten(),
                'RMSE': [rmse_CNN] * len(predictionsCNN),  # Repeat RMSE for all rows
                'MAE': [mae_CNN] * len(predictionsCNN),  # Repeat MAE for all rows
                'MAPE': [mape_CNN] * len(predictionsCNN)
            })
            df_output.to_excel(writer, sheet_name=sheet_name, index=False)
