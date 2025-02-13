#!/bin/bash

#SBATCH --job-name=PLM_embeddings  
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=20           
#SBATCH --gres=gpu:4               
#SBATCH --partition=scavenger-gpu,gpu-common,singhlab-gpu,cellbio-dgx
#SBATCH --mem=200G 
#SBATCH --output=./runs/logs/run_embedder_%j.out  
#SBATCH --error=./runs/logs/run_embedder_%j.err 

# Set environment variables for PyTorch DDP
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=4
export OMP_NUM_THREADS=20
export NCCL_DEBUG=INFO


torchrun --nnodes=1 --nproc_per_node=2 src/embeddings/generate.py --model_capacity 8M --dataset_file toy_set_10kseqs.fasta --batch_size 64
torchrun --nnodes=1 --nproc_per_node=2 src/embeddings/generate.py --model_capacity 150M --dataset_file toy_set_10kseqs.fasta --batch_size 64
torchrun --nnodes=1 --nproc_per_node=2 src/embeddings/generate.py --model_capacity 650M --dataset_file toy_set_10kseqs.fasta --batch_size 16
torchrun --nnodes=1 --nproc_per_node=2 src/embeddings/generate.py --model_capacity 3B --dataset_file toy_set_10kseqs.fasta --batch_size 8


