from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
import numpy as np
import matplotlib.pyplot as plt

import json
import time
import warnings

from quantum import *

#BUILD QUANTUM CIRCUIT
InputStateDimension = 5
EncodedStateDimension = 3
Depth = 3

Encoder = EncoderBuilder(InputStateDimension, EncodedStateDimension, Depth)
Circuit = QuantumCircuit(Encoder.num_qubits, Encoder.num_clbits)
for i in range(Circuit.num_qubits):
    Circuit.reset(i)
for i in range(2,5,1):
    Circuit.x(i)
Circuit.compose(Encoder, inplace=True)
Circuit.draw('mpl', filename='circuit.png')

# Define your cost function
def cost_function(params_values):
    #Update parameters in circuit
    params = dict(zip(Circuit.parameters, params_values))
    Circuit.assign_parameters(params, inplace=True)
    #print('Parameters in quantum circuit: ',params_values)

    # Transpile for simulator
    simulator = Aer.get_backend('qasm_simulator')
    circ = transpile(Circuit, simulator)

    # Run and get counts
    result = simulator.run(circ,shots=1000).result()
    counts = result.get_counts(circ)
    
    # Calculate the probability of getting measurement outcome '1' on the last qubit
    try:
        prob = counts['1']/1000
    except:
        prob = 0
    
    cost = 1 - prob

    print('Cost function value:', cost)

    return cost

# Initialize the COBYLA optimizer
opt = COBYLA()
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


