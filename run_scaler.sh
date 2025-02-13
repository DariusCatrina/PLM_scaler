#!/bin/bash

#SBATCH --job-name=PLM_scaler
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=20           # Number of CPU cores per task
#SBATCH --partition=scavenger,common,singhlab,cellbio-dgx
#SBATCH --mem=140G 
#SBATCH --output=./runs/logs/run_scaler_%j.out  
#SBATCH --error=./runs/logs/run_scaler_%j.err 

export OMP_NUM_THREADS=20

# Run the script
python src/scaler/scaler.py
