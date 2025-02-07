from embed_dataset import ESMDataset
from util import *
from config import *
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# _, alphabet, _ = get_model_tuple('8M')
# batch_converter = alphabet.get_batch_converter()

# dataset = ESMDataset(batch_converter=batch_converter, seq_file='/hpc/group/singhlab/rawdata/uniref50/toy_set.fasta')

# seq_len, seq_name, seq_tok = dataset[3]

def run_embedding_generation(rank, world_size, dataset):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # model.to(rank)
    # model = DDP(model, device_ids=[rank])
    # model.eval()

    data_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False 
    )

    embedding_loader = DataLoader(
        dataset,
        batch_size=32, 
        sampler=data_sampler,
        num_workers=8,
        pin_memory=True
    )
    if dist.get_rank() == 0:
        print('********* Sanity check *********')
        print(f'On device {rank}')
        print(f'Length of dataset: {len(dataset)}, batch_size: {32}')
        print(f'Dataloader length: {len(embedding_loader)}')
        print('*********')
    dist.destroy_process_group()

def main(n_gpus, model_capacity, dataset_file):
    world_size = n_gpus

    
    print(f'Loading the {model_capacity} model...')
    model, alphabet, repr_layer = get_model_tuple(model_capacity)
    batch_converter = alphabet.get_batch_converter()

    
    print('Loading the dataset...')
    dataset = ESMDataset(batch_converter=batch_converter, seq_file=dataset_file)

    mp.spawn(
        run_embedding_generation,
        args=(world_size, dataset,),
        nprocs=world_size,
        join=True)


if __name__ == "__main__":

    ### TODO: add it from arg ###
    n_gpus = 2
    model_capacity = '8M'
    dataset_file = '/hpc/group/singhlab/rawdata/uniref50/toy_set_1000seqs.fasta'

    mp.set_start_method("spawn", force=True)
    main(n_gpus=n_gpus, model_capacity=model_capacity, dataset_file=dataset_file)
    



