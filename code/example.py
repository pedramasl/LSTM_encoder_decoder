# Author: Laura Kulowski

'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import sys

import generate_dataset
import lstm_encoder_decoder
import plotting 

matplotlib.rcParams.update({'font.size': 17})


pvCycles = generate_dataset.synthetic_data()
train_split, test_split = generate_dataset.train_test_split(pvCycles, split = 0.8)
y_all = np.concatenate(pvCycles)
t_all = np.arange(0, len(y_all))
yy_train = np.concatenate(train_split)
yy_test = np.concatenate(test_split)
tt_train = np.arange(0, len(yy_train))
tt_test = np.arange(len(yy_train), len(yy_train) + len(yy_test))
plt.figure(figsize = (18, 6))
plt.plot(t_all, y_all, color = 'k', linewidth = 2)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.title('Temp Time Series')
plt.savefig('code/plots/temp_time_series.png')
# train_split = np.concatenate(train_split)
# test_split = np.concatenate(test_split)
# t_train = np.arange(0, len(train_split))
# y_train = train_split[0]
# t_test = np.arange(len(train_split), len(train_split) + len(test_split))
# y_test = test_split[0]
plt.figure(figsize = (18, 6))
plt.plot(tt_train, yy_train, color = '0.4', linewidth = 2, label = 'Train') 
plt.plot(np.concatenate([tt_train, tt_test]), np.concatenate([yy_train, yy_test]),
         color = (0.74, 0.37, 0.22), linewidth = 2, label = 'Test')
plt.ylabel(r'$y$')
plt.title('Time Series Split into Train and Test Sets')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout
plt.savefig('code/plots/train_test_split.png')
#----------------------------------------------------------------------------------------------------------------
# window dataset

# set size of input/output windows 
iw = 100
ow = 1
# s = 1
# trainsplit = np.concatenate(train_split)
# generate windowed training/test datasets
# Xtrain, Ytrain= generate_dataset.windowed_dataset(trainsplit, input_window = iw, output_window = ow, stride = s)
X_train, Y_train = generate_dataset.create_sequences(train_split, iw, ow, True)
# Xtest, Ytest = generate_dataset.windowed_dataset(test_split, input_window = iw, output_window = ow, stride = s)
X_test, Y_test = generate_dataset.create_sequences(test_split, iw, ow, True)
# plot example of windowed data  
plt.figure(figsize = (10, 6)) 
plt.plot(np.arange(0, iw), X_train[:, 0, 0], 'k', linewidth = 2.2, label = 'Input')
plt.plot(np.arange(iw - 1, iw + ow), np.concatenate([[X_train[-1, 0, 0]], Y_train[:, 0, 0]]),
         color = (0.2, 0.42, 0.72), linewidth = 2.2, label = 'Target')

plt.xlim([0, iw + ow - 1])
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.title('Example of Windowed Trainigoong Data')
plt.legend(bbox_to_anchor=(1.3, 1))
plt.tight_layout() 
plt.savefig('code/plots/windowed_data.png')

# convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(X_train, Y_train, X_test, Y_test)


# specify model parameters and train
model = lstm_encoder_decoder.lstm_seq2seq(input_size = 1, hidden_size = 100)
loss = model.train_model(X_train, Y_train, n_epochs = 200, target_len = ow, batch_size = 128, training_prediction = 'teacher_forcing', teacher_forcing_ratio = 0.9, learning_rate = 0.01, dynamic_tf = False)

plotting.plot_train_test_results(model, X_train, Y_train, X_test, Y_test)
plt.close('all')

