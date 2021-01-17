import os
import torch as T
import torch.nn as nn

from torch.distributions.normal import Normal
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn import Module
from torch.nn.modules import activation


class Net(nn.Module):
    def __init__(self, lr, input_dims, fc2_dims, n_actions):
        super(Net, self).__init__()
        self.input_dims = input_dims,
        self.fc2_dims = fc2_dims,
        self.n_actions = n_actions,
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # negtive log likelihood
        self.loss = nn.NLLLoss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        y = F.relu(self.fc2(x))
        actions = self.fc3(y)

        return actions


#a keras dense layer has two components: kernel matrix and bias vector
#lern distributon over weights

#bayesian weight uncertainty: tfp.layers.densevariations - unknown unknowns: epistemic uncertainty; not sure what data is not telling me

#https://www.youtube.com/watch?v=X6pCO0-HYVE
'''
def net():
 model = keras.Sequential([
        tfp.layers.DenseVariational(64, activtion='relu', input_shape = [len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model
'''

