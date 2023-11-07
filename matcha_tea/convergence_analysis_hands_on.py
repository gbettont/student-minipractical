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
Convergence analysis
====================

We show in this example the creation and run of a random quantum circuit.
In particular, we show the Quantum Neural Network circuit studied by
`Abbas et al <https://doi.org/10.1038/s43588-021-00084-1>`_.
Random circuit creates a big amount of entanglement: indeed, we will see
that to obtain a relaiable result we need a big enough maximum bond dimension,
the parameter that controls the amount of entanglement we can describe in the
MPS. The focus of this example is, however, to show how the convergence
parameters influences the evolution of the MPS system when the truncation
matters.

Try to answer the following questions:
1) What is the minimum bond dimension needed to fully represent your circuit?
2) Does it scale with the number of qubits?
3) How is the convergence parameter `cut_ratio` involved? Try to change it a bit
4) What is the probability distribution of the output bitstrings of your system?
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
import qtealeaves.observables as obs
from qmatchatea import run_simulation, QCConvergenceParameters, QCBackend
from qmatchatea.py_emulator import QCEmulator

###############################################################################
# Preparing the circuit. To see the exact structure of the circuit print it
# or refer to the paper.
# We use a linear entanglement structure, i.e. qubits talk only with linear
# nearest neighbors. Furthermore, to ensure an highly entangled state we
# select a number of repetitions equal to the number of qubits.
# Finally, we draw the necessary number of random parameters and assign
# them to the circuit.

num_qubits = 10
plot = False
qc = ... # Create your own quantum circuit with a lot of entanglement


###############################################################################
# Define the observables we are interested in. In this case we are interested
# in the final statevector, so we save the state, and also we want to ensure
# the final state is random, and so we want to check if the final parity of
# the state is similar to 0. The parity is a :math:`\sigma_z` pauli matrix
# applied to each site of the state. Since it is a new operator, we need
# to add it to the operators class.
# Since we expect an high entanglement, we also measure the bond entropy
# of the system

observables = obs.TNObservables()
observables += ... # Observable for saving the file
observables += ... # Add the observable for the projective measurements! What is the probability distribution of your circuit?

###############################################################################
# Define the convergence parameters, in particular the maximum bond dimension.
# The maximum bond dimension controls how much entanglement we can encode in
# the system while still obtaining trustful results.
# The maximum bond dimension reachable for a given number of qubits :math:`n`
# is :math:`\chi=2^{\lfloor\frac{n}{2}\rfloor}`. As you see, if we want to
# encode *any* state we still have an exponential scaling.
# However, we have ways to understand if the results of our computations are
# meaningful, through the analysis of the singular values cut through the
# simulation. So, we perform different simulations for an increasing bond
# dimension and check the results with a bond dimension we know is enough
# for obtaining convergent results.
# To perform this check we compute the fidelity of the state, i.e.
# :math:`\mathcal{F}=|\langle\psi_\chi|\psi_{\chi_{max}}\rangle|^2`, and then
# inspect the results on a log-log scale, plotting the infidelity, i.e.
# :math:`1-\mathcal{F}`.
# The important point is that we don't need knowledge about the statevector
# for computing the scalar product: it is rigorously defined also between
# MPS. The method used for that is :func:`contract`

bond_dims = np.unique(np.logspace(0, num_qubits // 2 + 1, 20, base=2, dtype=int))
fidelities = np.zeros(len(bond_dims))
singvals = []
backend = QCBackend(
    ... # Select the backend, and maybe try to see if something changes with the precision
)

for idx, chi in enumerate(bond_dims[::-1]):
    # Define the convergence parameters for each iteration
    conv_params = ...
    # Run the simulation on the python backend
    res = ...

    # Special case for the highest bond dimension investigated
    if idx == 0:
        reference_mps = QCEmulator.from_tensorlist(res.tens_net, conv_params)

    # Saving the singular values cut in the simulation from the res variable
    singvals.append( ... )

    # Using the state saved using the observable TNState2File initialize
    # an MPS and then compute the scalar product
    mps_state = QCEmulator.from_tensorlist(res.tens_net, conv_params)
    fidelities[idx] = (np.abs(reference_mps.emulator.contract(mps_state.emulator))) ** 2

###############################################################################
# Plotting the results. First, we notice that we reached convergence. This is
# showed by the plateau to the right of the plot: for a high-enough bond
# dimension increasing it doesn't affect the system at all. This behavior can
# be encountered also when we are below the maximum bond dimension for a given
# number of sites :math:`\chi_{max}=d^{n/2}`, and it means that the bond dimension
# is sufficient to capture the true behavior of the system.
infidelities = 1 - fidelities[::-1]
if plot:
    plt.plot(bond_dims, infidelities, "o--", color="blue")
    plt.loglog()
    plt.xlabel("Bond dimension $\\chi$", fontsize=14)
    plt.ylabel(
        "Infidelity $1-|\\langle\\psi_\\chi|\\psi_{\\chi_{max}}\\rangle|^2$",
        fontsize=14,
    )
    plt.title("Infidelity evolution with the bond dimension", fontsize=16)
    plt.savefig("infidelity.pdf")

###############################################################################
# Here instead we show the evolution of the cumulative truncated norm. At each
# two-qubit gate, when the bond dimension is not enough for correctly describe
# the system, we are truncating the norm of the state. Along the evolution we
# keep track of the truncation. In the plot, we can observe how the norm
# truncated decreases as we increase the bond dimension. However, only the
# highest bond dimension possible gives 0 truncation.
ii = 0
for singv, bd in zip(singvals[::-1], bond_dims):
    if ii % 5 == 0:
        plt.plot(np.cumsum(singv), ".--", label=f"$\\chi$={bd}")
    ii += 1

if plot:
    plt.xlabel("Number of two-qubit gates", fontsize=14)
    plt.ylabel("Cumulative norm truncated", fontsize=14)
    plt.title("Norm truncated for different bond dimensions", fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("infidelity.pdf")

