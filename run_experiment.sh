#!/bin/bash

#SBATCH --job-name=experiment
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=20          # Number of CPU cores per task
#SBATCH --partition=scavenger,common,singhlab,compalloc
#SBATCH --mem=120G 
#SBATCH --output=./runs/logs/run_experiment_%j.out  
#SBATCH --error=./runs/logs/run_experiment_%j.err 

export OMP_NUM_THREADS=20

set -e

# torchrun --nnodes=1 --nproc_per_node=4 -m src.mutation_experiment.generate --model_capacity 8M --dataset_file BRCA1_HUMAN_Findlay_2018.csv --batch_size 4 &&
# torchrun --nnodes=1 --nproc_per_node=4 -m src.mutation_experiment.generate --model_capacity 150M --dataset_file BRCA1_HUMAN_Findlay_2018.csv --batch_size 4 &&
# torchrun --nnodes=1 --nproc_per_node=4 -m src.mutation_experiment.generate --model_capacity 650M --dataset_file BRCA1_HUMAN_Findlay_2018.csv --batch_size 4 &&
python -m src.mutation_experiment.mutation_experiment