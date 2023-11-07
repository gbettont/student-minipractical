# This code is part of qmatchatea.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
Quantum Fourier Transform example
=================================

This example show the application of a quantum Fourier transform on the
:math:`|++++++++++\rangle` state, with
:math:`|+\rangle=\sqrt{2}^{-1}(|0\rangle+|1\rangle`. The choice of this
initial state is particularly interesting for testing porpouses, since
its Fourier transform is simply :math:`|0000000000\rangle`, as we will
see at the end of the example. Furthermore, the entanglement created
by the circuit is very small: thus you can increase the number of
qubits `num_qubits` to hundreds without affecting too much the computational
time.



Try to answer the following questions:
1) What is the minimum bond dimension to get the correct result?
2) Try to use the qiskit implementation for the QFT. Is the bond dimension needed different?
3) Does the bond dimension needed in either of the two cases scale with the number of qubits?
"""

from qiskit import QuantumCircuit
import qtealeaves.observables as obs

from qmatchatea.qk_utils import QFT_qiskit
from qmatchatea import run_simulation, QCConvergenceParameters, QCBackend

###############################################################################
# Preparing the circuit

num_qubits = 10
qc = QuantumCircuit(num_qubits)

###############################################################################
# We apply an hadamard to all the qubits, such
# that the output after the QFT will be a single
# state rather than a superposition

for ii in range(num_qubits):
    qc.h(ii)

###############################################################################
# Apply a nearest-neighbors version of the QFT,
# optimized for an MPS simulation

_ = QFT_qiskit(qc, num_qubits)

###############################################################################
# Define the observables. In this example we are interested in:
#
# - Projective measurement, i.e. a defined number of final projective measurement
#   performed on the circuit after the evolution. These are snapshots of the
#   final quantum states. With a sufficient number of shots one can reconstruct
#   the statistics of the state
# - Saving the state for further use. We save the **mps** state, which contains
#   a lot of information. In particular, for a reduced number of qubits, we can
#   then get the full statevector of the quantum state.

observables = obs.TNObservables()
# We want final projective measurements
observables += obs.TNObsProjective(1024)

# We also want to save the state for further use. To use the result in python,
# we set the formatting to formatted, i.e. 'F'
observables += obs.TNState2File("state.txt", "F")

###############################################################################
# We set the convergence parameters for the MPS simulation

conv_params = QCConvergenceParameters(max_bond_dimension=...)

###############################################################################
# It is possible to choose between different approaches for running the simulation:
# - The backend can be either python "PY" or fortran "FR".
# - The machine precision can be either double complex "Z" or single complex "C".
# - The device of the simulation can be either "cpu" or "gpu".
# - The number of processes for the MPI simulation. If 1 is passed, then a serial
#   program is run.
# - The approach for the MPI simulation. Either master/worker "MW", cartesian "CT" or
#   serial "SR".
#

backend = QCBackend(
    ... # Fill in the backend!
)

results = run_simulation(
    ...
)

###############################################################################
# We finally print the results, obtaining as expected the
# :math:`|0000000000\rangle` state.

for key, value in results.measures.items():
    print(f"{key} : {value}")

