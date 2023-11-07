#!/bin/bash
#SBATCH -A #REPLACEME
#SBATCH -p boost_usr_prod
#SBATCH --time 00:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:#REPLACEME        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB 
#SBATCH --job-name=#REPLACEME
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err


python solving_MSI_problem.py 
