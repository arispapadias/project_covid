import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.signal import savgol_filter
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn import BCELoss
from torch.nn.init import xavier_uniform_
from numpy import sqrt
from numpy import vstack
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

list = []
i = 0
for i in range(1000):
    list.append([i+1])


list = np.array(list, dtype=float)
data = list[:-1]

# plt.plot(data)
# plt.show()

class scalerClass(object):
    def __init__(self, scaler_type, data):
        self.scaler_type = scaler_type
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def scale(self, data):
        # Input data shape [K, D]
        assert len(np.shape(data))==2
        # X' = (X - μ) / σ
        if self.scaler_type == "standard":
            # Adding the dimension for the batch
            mean = self.mean[np.newaxis]
            std  = self.std[np.newaxis]
            data = (data-mean) / std
        # X' = (X - Xmin) / (Xmax - Xmin) 
        elif self.scaler_type == "minMaxZeroOne":
            # Adding the dimension for the batch
            min_ = self.min[np.newaxis]
            max_  = self.max[np.newaxis]
            data = (data - min_) / (max_ - min_)
        return data


data_after = data
data_after = np.reshape(data_after, (-1, 1))
scaler_type = "minMaxZeroOne"
scaler = scalerClass(scaler_type, data)
# print("Before scaling:")
# print(np.mean(data))
# print(np.std(data))
data = scaler.scale(data)
# print(data)
print("After scaling:")
# print(np.mean(data))
# print(np.std(data))
print(np.min(data))
print(np.max(data))

timesteps_input = 1
timesteps_output = 1

num_timesteps = len(data)
print("Number of timesteps = {:}".format(num_timesteps))
# Form the data batches
max_samples = num_timesteps - timesteps_input # - timesteps_output
print("Maximum number of samples = {:}".format(max_samples))

# 1, 2, 3, 4, 5 -> 6
# 2, 3, 4, 5, 6 -> 7
# 3, 4, 5, 6, 7 -> 8
samples_input = []
samples_target = []
for sample_in in range(max_samples):
    sample_input = data[sample_in:sample_in+timesteps_input]
    sample_target = data[sample_in+timesteps_input:sample_in+timesteps_input+timesteps_output]
    # print(sample_in)
    # print(np.shape(sample_input))
    # print(np.shape(sample_target))
    samples_input.append(sample_input)
    samples_target.append(sample_target)

samples_input = np.array(samples_input)
samples_target = np.array(samples_target)
print("Samples:")
num_samples = len(samples_input)
# print(np.shape(samples_input))
# print(np.shape(samples_target))


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, n_output):
        super(MLP, self).__init__()
        # input to first  layer
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second layer
        self.hidden2 = Linear(10, 10)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third layer
        self.hidden3 = Linear(10, 10)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        # forth layer
        self.hidden4 = Linear(10, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()
        # fifth layer
        self.hidden5 = Linear(10, 10)
        xavier_uniform_(self.hidden5.weight)
        self.act5 = Sigmoid()
        # sixth layer and output
        self.hidden6 = Linear(10, n_output)
        xavier_uniform_(self.hidden6.weight)

    def forward(self, X): # [K, n_inputs]
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # forth hidden layer
        X = self.hidden4(X)
        X = self.act4(X)
        # fifth hidden layer
        X = self.hidden5(X)
        X = self.act5(X)
        # sixth hidden layer
        X = self.hidden6(X)
        return X

    def forwardAndComputeLoss(self, input_, target_):
        output_ = self.forward(input_)
        loss = ((target_-output_)**2).mean()
        return output_, loss


network = MLP(timesteps_input, timesteps_output)

K = 32  # BATCH SIZE
input_data = samples_input[0:K, :, 0]
print(input_data)

input_data = torch.tensor(samples_input).float()
output = network.forward(input_data)

# print(output.size())
# print(output)

idx_shuffle = np.random.permutation(len(samples_input))
samples_input   = samples_input[idx_shuffle]
samples_target  = samples_target[idx_shuffle]

N_train = int(num_samples*0.8)
print("Training samples {:}/{:}.".format(N_train, num_samples))

samples_input_train   = samples_input[:N_train]
samples_target_train  = samples_target[:N_train]
samples_input_val   = samples_input[N_train:]
samples_target_val  = samples_target[N_train:]
# print("Training data shapes:")
# print(np.shape(samples_input_train))
# print(np.shape(samples_target_train))
# print(np.shape(samples_input_val))
# print(np.shape(samples_target_val))


""" dataset and dataload """

batch_size=32

from torch.utils.data import TensorDataset, DataLoader

samples_input_train = torch.Tensor(samples_input_train)
samples_target_train = torch.Tensor(samples_target_train)

samples_input_val = torch.Tensor(samples_input_val)
samples_target_val = torch.Tensor(samples_target_val)

dataset_train = TensorDataset(samples_input_train, samples_target_train)
dataset_val = TensorDataset(samples_input_val, samples_target_val)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

print("Dataset sizes:")
print(len(dataloader_train))
print(len(dataloader_val))

""" Let's try to plot some samples """
samples_input_val = torch.Tensor(samples_input_val)
samples_target_val = torch.Tensor(samples_target_val)



def train_model(train_dl, network):
    # define the optimization
    # criterion = BCELoss()
    criterion = MSELoss()
    optimizer = SGD(network.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 100
    """ Create PROGRESS bar plot """
    progress_bar = tqdm(total=num_epochs)
    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            output = network(inputs)
            # calculate loss
            loss = criterion(output, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        progress_bar.update(1)
    progress_bar.close()

# axiologisi to dataloader_val
def evaluate_model(test_dl, network):
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        output = network(inputs)
        # retrieve numpy array
        output = output.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        output = output.reshape((len(output), 1))
        # print(np.shape(actual))
        # print(np.shape(output))
        # print(ark)
        # store
        predictions.append(output)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    # np.mean((actuals- predictions)**2)
    return mse



train_model(dataloader_train, network)
evaluate_model(dataloader_val, network)


