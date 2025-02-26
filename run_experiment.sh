#!/bin/bash

#SBATCH --job-name=PLM_embeddings  
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=40          
#SBATCH --gpus-per-node=4            
#SBATCH --partition=scavenger,common,singhlab,cellbio-dgx
#SBATCH --mem=210G 
#SBATCH --output=./runs/logs/run_embedder_%j.out  
#SBATCH --error=./runs/logs/run_embedder_%j.err 

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=4
export OMP_NUM_THREADS=40
export NCCL_DEBUG=INFO

set -e

torchrun --nnodes=1 --nproc_per_node=4 -m src.mutation_experiment.generate --model_capacity 8M --dataset_file BRCA1_HUMAN_Findlay_2018.csv --batch_size 512 &&
torchrun --nnodes=1 --nproc_per_node=4 -m src.mutation_experiment.generate --model_capacity 150M --dataset_file BRCA1_HUMAN_Findlay_2018.csv --batch_size 512 &&
torchrun --nnodes=1 --nproc_per_node=4 -m src.mutation_experiment.generate --model_capacity 650M --dataset_file BRCA1_HUMAN_Findlay_2018.csv --batch_size 512 &&
torchrun --nnodes=1 --nproc_per_node=4 -m src.mutation_experiment.generate --model_capacity 3B --dataset_file BRCA1_HUMAN_Findlay_2018.csv --batch_size 64 