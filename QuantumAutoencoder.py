#-------------------------------------------#
#           QuantumAutoencoder              #
#-------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


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

#LOAD OPTIONS FROM FILE
with open("./options.json", "r") as file:
    namelist = json.load(file)

depth = int(namelist["DEPTH"])
ITERATIONS = int(namelist["ITERATIONS"])
MODE = str(namelist["MODE"])
NOISY = bool(namelist["NOISY"])
print("Option namelist loaded")

#--------------------------------------------

if MODE == "TRAIN" and NOISY:
    warnings.warn("NOISY is set to True during training, this will affect the optimization process")
    print("Do you want to continue? [y/n]")
    if input() != "y" or input() != "Y":
        exit()

print("Mode:", MODE)
print("Noisy:", NOISY)
if MODE == "TRAIN":
    print("Depth:", depth)
    print("Iterations:", ITERATIONS)
    print("Do you want to continue? [y/n]")
    if input() != "y" or input() != "Y":
        exit()
print()

#--------------------------------------------

def Identity(x):
    return x

#--------------------------------------------

def cost_func_digits(params_values):
    probabilities = qnn.forward(train_images, params_values)
    cost = np.sum(probabilities[:, 1]) / train_images.shape[0]

    print(f"Iteration {len(objective_func_vals)}: {cost}")

    # plotting part
    clear_output(wait=True)
    objective_func_vals.append(cost)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)

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

fm = RawFeatureVector(2 ** (num_latent + num_trash))
ae = EncoderBuilder(num_latent+num_trash, num_latent, depth)
qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
qc = qc.compose(fm, range(num_latent + num_trash))
qc = qc.compose(ae)

#--------------------------------------------

#SETUP QNN
qnn = SamplerQNN(
    circuit=qc,
    input_params=fm.parameters,
    weight_params=ae.parameters,
    interpret=Identity,
    output_shape=2,
)

#--------------------------------------------

#OPTIMIZATION
if MODE == "TRAIN":
    print("Optimization started:")
    opt = COBYLA(maxiter=ITERATIONS, disp=True)
    objective_func_vals = []
    initial_point = np.random.rand(ae.num_parameters)
    plt.rcParams["figure.figsize"] = (12, 6)
    start = time.time()
    opt_result = opt.minimize(fun=cost_func_digits, x0=initial_point, bounds=[(0, 2 * np.pi)] * ae.num_parameters)
    elapsed = time.time() - start

    parameters = opt_result.x
    print(f"Fit in {elapsed:0.2f} seconds")
    plt.show()

#SAVE PARAMETERS TO FILE
    with open("./OPTIMIZATION/parameters.json", "w") as file:
        json.dump(list(opt_result.x), file)
    print("Parameters saved to file: ./OPTIMIZATION/parameters.json")
    print()

#--------------------------------------------
    
#SETUP DECODER CIRCUIT
Decoder = QuantumCircuit(num_latent + num_trash)
Decoder = Decoder.compose(fm)
ansatz_qc = AnsatzBuilder(num_latent + num_trash, depth)
Decoder = Decoder.compose(ansatz_qc)
Decoder.barrier()
Decoder.reset(4)
Decoder.reset(3)
Decoder.barrier()
Decoder = Decoder.compose(ansatz_qc.inverse())

#LOAD PARAMETERS FROM FILE
if MODE == "TEST":
    print("Loding parameters from file: ./OPTIMIZATION/parameters.json")
    with open("./OPTIMIZATION/parameters.json", "r") as file:
        parameters = json.load(file)
    print()

#--------------------------------------------
#TESTING
        
test_images, test_labels = GetDatasetDigits(2, draw=False)
i = 0
for image, label in zip(test_images, test_labels):

    #ADD NOISE
    if NOISY and MODE == "TEST":
        noise = np.random.uniform(0, 0.4, size=32)
        original_image = image
        image = image + noise
        image = image / np.linalg.norm(image)
        print("Norm of noisy image:", np.linalg.norm(image))

    param_values = np.concatenate((image, parameters))
    output_qc = Decoder.assign_parameters(param_values)
    output_sv = Statevector(output_qc).data
    output_sv = np.reshape(np.abs(output_sv) ** 2, (8, 4))
    output_sv = output_sv / np.linalg.norm(output_sv)

    if NOISY and MODE == "TEST":
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(original_image.reshape(8, 4))
        ax1.set_title("Original Data")
        ax2.imshow(image.reshape(8, 4))
        ax2.set_title("Noisy Data")
        ax3.imshow(output_sv)
        ax3.set_title("Output Data")
        plt.savefig(f'./DATA/{1-i}_image.png')
        print(f'Image saved to file: ./DATA/{1-i}_image.png')

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image.reshape(8, 4))
        ax1.set_title("Input Data")
        ax2.imshow(output_sv)
        ax2.set_title("Output Data")
        plt.savefig(f'./DATA/{1-i}_image.png')
        print(f'Image saved to file: ./DATA/{1-i}_image.png')

    #SCRIVO SU FILE PER SUCCESSIVE ANALISI
    if NOISY and MODE == "TEST":
        np.savetxt(f'./DATA/{1-i}_input.txt', original_image.reshape(8, 4), fmt='%f')
        np.savetxt(f'./DATA/{1-i}_noisy.txt', image.reshape(8, 4), fmt='%f')
        np.savetxt(f'./DATA/{1-i}_output.txt', output_sv, fmt='%f')
        print(f'Data saved to file: ./DATA/{1-i}_input.txt')
        print(f'Data saved to file: ./DATA/{1-i}_noisy.txt')
        print(f'Data saved to file: ./DATA/{1-i}_output.txt')
    else:
        np.savetxt(f'./DATA/{1-i}_input.txt', image.reshape(8, 4), fmt='%f')
        np.savetxt(f'./DATA/{1-i}_output.txt', output_sv, fmt='%f')
        print(f'Data saved to file: ./DATA/{1-i}_output.txt')
        print(f'Data saved to file: ./DATA/{1-i}_input.txt')

    i = i+1
