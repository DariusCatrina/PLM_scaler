#!/bin/bash

#SBATCH --job-name=PLM_embeddings  
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH --partition=scavenger-gpu,gpu-common,singhlab-gpu,cellbio-dgx
#SBATCH --mem=10G 
#SBATCH --output=./runs/logs/run_%j.out  
#SBATCH --error=./runs/logs/run_%j.err 

# Set environment variables for PyTorch DDP
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=2
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

# Run the script
torchrun --nnodes=1 --nproc_per_node=2 src/embeddings/generate.py

