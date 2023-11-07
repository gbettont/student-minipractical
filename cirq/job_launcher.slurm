#!/bin/bash
#SBATCH -A #REPLACEME
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=#REPLACEME # 4 tasks out of 32
#SBATCH --gres=gpu:#REPLACEME        # #REPLACEME gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB 
#SBATCH --job-name=cirq_job
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err

nvidia-smi -l &> nvidiasmi.log &

#python simulate_a_large_quantum_circuit.py
#python test_gpu.py
#python ghz.py
#python quantum_fourier_transform.py
#python qaoa_max-cut.py
python scalability.py
