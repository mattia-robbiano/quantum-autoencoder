import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN 
from myquantum import *
from qiskit_algorithms.optimizers import COBYLA

from IPython.display import clear_output
import json
import time
import warnings

#PARAMETRI
q = [0,0,1,1,1,0,0,0,0,0]
EncodedStateDimension = 3
Depth = 3

#BUILD ENCODER CIRCUIT
InputStateDimension = len(q)
Encoder = EncoderBuilder(InputStateDimension, EncodedStateDimension, Depth)
Circuit = QuantumCircuit(Encoder.num_qubits, Encoder.num_clbits)
#(Reset all qubits to 0 and change to 1 through x gate qubit to get the state q)
for i in range(Circuit.num_qubits):
    Circuit.reset(i)
for i in range(InputStateDimension):
    if q[i] == 1:
        Circuit.x(i)
Circuit.compose(Encoder, inplace=True)

# INTERFACE CIRCUIT-SIMULATOR SamplerQNN
def identity_interpret(x):
    return x
qnn = SamplerQNN(
    circuit=Circuit, #circuito completo
    input_params=[], #guess iniziale lasciato a SamplerQNN, quindi randomici
    weight_params=Circuit.parameters, #identificato i parametri del circuito, quindi i parametri di RealAmplitudes
    interpret=identity_interpret,
    output_shape=2, #l'output del circuito può essere 1 o 0, una stringa binaria di dimensione 2
)

#COST FUNCTION
def cost_function(params_values):
    probabilities = qnn.forward([], params_values)
    cost = np.sum(probabilities[:, 1])
    
    # update plot and outputs
    clear_output(wait=True)
    objective_func_vals.append(cost)
    print(len(objective_func_vals))
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    return cost
    
objective_func_vals = []

# Initialize the COBYLA optimizer
opt = COBYLA(maxiter=50)
num_parameters = Circuit.num_parameters
initial_point = np.random.rand(num_parameters)  # Set the initial parameters
print('Number of parameters in quantum circuit: ',Circuit.num_parameters)
print('Initial parameters in quantum circuit: ',initial_point)

# Perform optimization
start = time.time()
opt_result = opt.minimize(cost_function,initial_point)
#print('Cost function value:', cost_function(initial_point))
elapsed = time.time() - start

print(f"Fit in {elapsed:0.2f} seconds")
plt.show()

