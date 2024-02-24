import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN 
from myquantum import *
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.circuit.library import RawFeatureVector

from IPython.display import clear_output
import json
import time
import warnings

#--------------------------------------------

def Identity(x):
    return x

qnn = SamplerQNN(
    circuit=qc,
    input_params=fm.parameters,
    weight_params=ae.parameters,
    interpret=Identity,
    output_shape=2,
)

#--------------------------------------------

def cost_func_digits(params_values):
    probabilities = qnn.forward(train_images, params_values)
    cost = np.sum(probabilities[:, 1]) / train_images.shape[0]

    # plotting part
    clear_output(wait=True)
    objective_func_vals.append(cost)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    return cost

#--------------------------------------------

def ZeroMask(j, i):
    # Index for zero pixels
    return [[i, j],[i - 1, j - 1],[i - 1, j + 1],[i - 2, j - 1],[i - 2, j + 1],[i - 3, j - 1],[i - 3, j + 1],[i - 4, j - 1],[i - 4, j + 1],[i - 5, j],
    ]


def OneMask(i, j):
    # Index for one pixels
    return [[i, j - 1], [i, j - 2], [i, j - 3], [i, j - 4], [i, j - 5], [i - 1, j - 4], [i, j]]

#--------------------------------------------

def GetDatasetDigits(num, draw=True):
    # Create Dataset containing zero and one
    train_images = []
    train_labels = []
    for i in range(int(num / 2)):

        # First we introduce background noise
        empty = np.array([algorithm_globals.random.uniform(0, 0.1) for i in range(32)]).reshape(8, 4) #original: 0.1
        # Now we insert the pixels for the one
        for i, j in OneMask(2, 6):
            empty[j][i] = algorithm_globals.random.uniform(0.9, 1) #original: 0.9
        train_images.append(empty)
        train_labels.append(1)
        if draw:
            plt.title("This is a One")
            plt.imshow(train_images[-1])
            plt.show()

    for i in range(int(num / 2)):
        # First we introduce background noise
        empty = np.array([algorithm_globals.random.uniform(0, 0.1) for i in range(32)]).reshape(8, 4)
        # Now we insert the pixels for the zero
        for k, j in ZeroMask(2, 6):
            empty[k][j] = algorithm_globals.random.uniform(0.9, 1)
        train_images.append(empty)
        train_labels.append(0)
        if draw:
            plt.imshow(train_images[-1])
            plt.title("This is a Zero")
            plt.show()

    train_images = np.array(train_images)
    train_images = train_images.reshape(len(train_images), 32)

    # Normalize the data
    for i in range(len(train_images)):
        sum_sq = np.sum(train_images[i] ** 2)
        train_images[i] = train_images[i] / np.sqrt(sum_sq)

    return train_images, train_labels


train_images, __ = GetDatasetDigits(2,False)


#--------------------------------------------


#SETUP ENCODER CIRCUIT
num_latent = 3
num_trash = 2
depth = 10

fm = RawFeatureVector(2 ** (num_latent + num_trash))
ae = EncoderBuilder(num_latent+num_trash, num_latent, depth)
qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
qc = qc.compose(fm, range(num_latent + num_trash))
qc = qc.compose(ae)

#OPTIMIZATION
opt = COBYLA(maxiter=500)
objective_func_vals = []
initial_point = np.random.rand(ae.num_parameters)  # Set the initial parameters
# make the plot nicer
plt.rcParams["figure.figsize"] = (12, 6)
start = time.time()
opt_result = opt.minimize(fun=cost_func_digits, x0=initial_point)
elapsed = time.time() - start

print(f"Fit in {elapsed:0.2f} seconds")
plt.show()

