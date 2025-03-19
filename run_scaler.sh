#!/bin/bash

#SBATCH --job-name=PLM_scaler
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=40          # Number of CPU cores per task
#SBATCH --partition=scavenger,common,singhlab,compalloc
#SBATCH --mem=400G 
#SBATCH --output=./runs/logs/run_scaler_%j.out  
#SBATCH --error=./runs/logs/run_scaler_%j.err 

export OMP_NUM_THREADS=40

python -m src.scaler.scaler --dataset_file toy_set_10Kseqs --model_capacity_in 150M --model_capacity_out 3B &&
python -m src.scaler.scaler --dataset_file toy_set_10Kseqs --model_capacity_in 650M --model_capacity_out 3B 


