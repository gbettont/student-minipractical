# Intro to Pennylane


## Create a circuit with 2 entangled qubits

# Import libraries
import pennylane as qml
from pennylane import numpy as np

# Create a device with 2 qubits
dev = qml.device('lightning.qubit', wires=2)

# Create a QNode with 2 entangled qubits
@qml.qnode(dev) # significa: qualunque cosa io scriva dopo questa riga, giralo sul device "dev"
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1]) # [control qubit,target qubit]
    return qml.probs() # probability measurement. Output [P(00),P(),P(),P(11)]

# Run the circuit
print(circuit())



## Create a parametrized circuit with 1 qubit

# Create a device with 1 qubit
dev2 = qml.device('default.qubit', wires=1)

# Create a parametrized circuit
@qml.qnode(dev2)
def circuit2(params):
    qml.RX(params[0],wires=0)
    qml.RY(params[1],wires=0)
    return qml.expval(qml.PauliZ(0))  # expectation value measurement, measured on the Z axes

# Run the circuit
print(circuit2([0.1,0.2]))

