# Import Cirq and qsim
import cirq
import qsimcirq

# Instantiate qubits and create a circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))

# Instantiate a simulator that uses the GPU
# xx = 0 for Option 1, 1 for Option 2, or the number of GPUs for Option 3.
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode = 4, max_fused_gate_size=4)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

# Run the simulation
print("Running simulation for the following circuit:")
print(circuit)

qsim_results = qsim_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01])

print("qsim results:")
print(qsim_results)
