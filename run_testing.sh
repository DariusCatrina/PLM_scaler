#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=10         
#SBATCH --partition=scavenger,common,singhlab,compalloc
#SBATCH --mem=150G 
#SBATCH --output=./runs/logs/run_testing_%j.out  
#SBATCH --error=./runs/logs/run_testing_%j.err 

export OMP_NUM_THREADS=10

python -m src.mutation_experiment.checker

