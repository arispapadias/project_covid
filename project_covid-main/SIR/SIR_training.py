import pickle
import numpy as np
import os

# import activations
import torch
import torch.nn as nn

from tqdm import tqdm

# from scaler import *
# from data import *
# from plotting import *


from scipy.integrate import odeint
import matplotlib.pyplot as plt

# I, R, S einai to STATE tou systimatos -> KATASTASI
# ta beta, gamma einai ta PARAMETERS -> PARAMETROUS
# PRIN: sto neuroniko, poio einai to STATE ? (sto MLP) X/Y
# PRIN: sto neuroniko, poia einai ta PARAMETERS ? (sto MLP) A kai b

# PRIN: eixame polla A, b, polla parameters ! (xiliades)
# TORA sto SIR: beta, gamma MONO 2. (mono 2)



# PRIN: montelo x_{t+1} = MLP(x_t)
# TORA: montelo x_{t+1} = integrator ( SIR (x_t))

# STOXOS: Na vroume tis parametrous tou montelou
# poy kanoune FIT sta data
# FIT ? to montelo na ta provlepei me mikro error
# Poies einai oi parametrous ? beta/gamma/I0 isos
# STEPS:
# 1. Pernoume dedomena apo Infected
# 2. STOXOS? to neuroniko na ta provlepsei !
# na vroume katallila beta/gamma/I0, klp pou na provlepoun ta dedomena

# NA VROUME DEDOMENA APO INFECTED



from data import *

data = getData()

print(np.shape(data))

# We assume that the inflected are proportional to deaths
N_greece = 20000
data = data/N_greece

# plt.plot(data)
# plt.show()

# we have data only from the infected, but we will train the NN.

from torch.nn import Parameter

class neuralSIR(nn.Module):
    def __init__(self):
        super(neuralSIR, self).__init__()
        self.beta = Parameter(torch.tensor(0.2))
        self.gamma = Parameter(torch.tensor(1./10.))
        self.N = N_greece
        print(f"Initialized neural SIR model with {self.beta} and {self.gamma}")

    def derivative(self, input_):
        # print(input_.size()) # [K, 3]
        S = input_[:, 0]
        I = input_[:, 1]
        R = input_[:, 2]

        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.gamma * I
        dRdt = self.gamma * I
        initial_state = torch.stack((dSdt, dIdt, dRdt), dim=1)
        return initial_state

    def forward(self, input_):
        dt = 1 # one day
        output = input_ + self.derivative(input_) * dt
        # self.derivative = (output - input_) / dt, protou vathmou approximation tou derivative
        # den einai akrivhs alla einai mia kalh arxh.
        return output

    def iterative(self, input_, horizon=1):
        dt = 1 # one day
        outputs = []
        for t in range(horizon):
            output = self.forward(input_)
            outputs.append(output)
        return outputs

model = neuralSIR()
print(model)
I0, R0 = 1, 0
S0 = N_greece - I0 - R0

I0 = torch.tensor(I0)
R0 = torch.tensor(R0)
S0 = torch.tensor(S0)
initial_state = torch.stack((S0, I0, R0))
# One batch
initial_state = torch.reshape(initial_state, (1, -1))
print(initial_state.size())

state_der = model.derivative(initial_state)
print(state_der.size())
next_state = model.forward(initial_state)
print(initial_state)
print(next_state)

evolution = model.iterative(initial_state, horizon=100)


# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()




print(ark)
# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10 
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(np.arange(len(data)), data, 'y', alpha=0.8, lw=2, label='Data')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


