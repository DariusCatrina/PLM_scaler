#!/bin/bash

#SBATCH --job-name=PLM_embeddings  
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=40          
#SBATCH --gpus-per-node=4            
#SBATCH --partition=scavenger-gpu,gpu-common
#SBATCH --mem=210G 
#SBATCH --output=./runs/logs/run_embedder_%j.out  
#SBATCH --error=./runs/logs/run_embedder_%j.err 

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=4
export OMP_NUM_THREADS=40
export NCCL_DEBUG=INFO

set -e

torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate.py --model_capacity 8M --dataset_file toy_set_10Kseqs.fasta --batch_size 128 &&
torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate.py --model_capacity 150M --dataset_file toy_set_10Kseqs.fasta --batch_size 64 &&
torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate.py --model_capacity 650M --dataset_file toy_set_10Kseqs.fasta --batch_size 32 &&
torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate.py --model_capacity 3B --dataset_file toy_set_10Kseqs.fasta --batch_size 4 
