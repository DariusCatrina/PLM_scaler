#!/bin/bash

#SBATCH --job-name=PLM_embeddings  
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=40          
#SBATCH --gpus-per-node=4             
#SBATCH --partition=cellbio-dgx
#SBATCH --mem=250G 
#SBATCH --output=./runs/logs/run_embedder_%j.out  
#SBATCH --error=./runs/logs/run_embedder_%j.err 

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=4
export OMP_NUM_THREADS=40
export NCCL_DEBUG=INFO

set -e

DATASET_NAME="UBR5_HUMAN_Tsuboyama_2023_1I2T"
mkdir ./data/eval/$DATASET_NAME
torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate_mut.py --model_capacity 8M --dataset_file ${DATASET_NAME}.csv --batch_size 8 &&
torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate_mut.py --model_capacity 150M --dataset_file ${DATASET_NAME}.csv --batch_size 8 &&
torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate_mut.py --model_capacity 650M --dataset_file ${DATASET_NAME}.csv --batch_size 4 &&
torchrun --nnodes=1 --nproc_per_node=4 src/embeddings/generate_mut.py --model_capacity 3B --dataset_file ${DATASET_NAME}.csv --batch_size 1

