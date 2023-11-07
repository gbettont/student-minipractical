import cirq
import qsimcirq
import time

n_qubits = 32
qubits = cirq.LineQubit.range(n_qubits)
circuit = cirq.Circuit()
circuit.append(cirq.H(qubits[0]))
circuit.append(cirq.CNOT(qubits[idx], qubits[idx + 1]) \
    for idx in range(n_qubits - 1))

s = cirq.sim.Simulator()

wall_start = time.monotonic()
process_start = time.process_time()

result = s.compute_amplitudes(circuit, [0, 2**n_qubits-1])

wall_end = time.monotonic()
process_end = time.process_time()

print(f'cirq.sim  : {result}')

total_process_time = process_end - process_start
total_wall_time = wall_end - wall_start

print(f"Wall Time: {total_wall_time}   -- Process Time: {total_process_time} ")


# qsim(CPU)
options = qsimcirq.QSimOptions(max_fused_gate_size=4, cpu_threads=512)
s = qsimcirq.QSimSimulator(options)

wall_start = time.monotonic()
process_start = time.process_time()

result = s.compute_amplitudes(circuit, [0, 2**n_qubits-1])

wall_end = time.monotonic()
process_end = time.process_time()

print(f'qsim(CPU) : {result}')
total_process_time = process_end - process_start
total_wall_time = wall_end - wall_start

print(f"Wall Time: {total_wall_time}   -- Process Time: {total_process_time} ")


# qsim(cuStateVec)
options = qsimcirq.QSimOptions(use_gpu=True, max_fused_gate_size=1, gpu_mode=1)
s = qsimcirq.QSimSimulator(options)

wall_start = time.monotonic()
process_start = time.process_time()

result = s.compute_amplitudes(circuit, [0, 2**n_qubits-1])

wall_end = time.monotonic()
process_end = time.process_time()
print(f'cuStateVec: {result}')
total_process_time = process_end - process_start
total_wall_time = wall_end - wall_start

print(f"Wall Time: {total_wall_time}   -- Process Time: {total_process_time} ")


