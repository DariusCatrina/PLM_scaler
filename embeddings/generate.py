from embed_dataset import ESMDataset
from util import *
from config import *
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def run_embedding_generation(rank, world_size, dataset):
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

    print('********* Sanity check *********')
    print(f'On device {rank}')
    print(f'Length of dataset: {len(dataset)}, batch_size: {32}')
    print(f'Dataloader length: {len(embedding_loader)}')
    print('*********')
    

def main(model_capacity, dataset_file):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if dist.get_rank() == 0:
        print(f'Loading the {model_capacity} model...')
    model, alphabet, repr_layer = get_model_tuple(model_capacity)
    batch_converter = alphabet.get_batch_converter()

    if dist.get_rank() == 0:
        print('Loading the dataset...')
    dataset = ESMDataset(batch_converter=batch_converter, seq_file=dataset_file)

    run_embedding_generation(rank, world_size, dataset)
    dist.destroy_process_group()


if __name__ == "__main__":

    ### TODO: add it from arg ###
    model_capacity = '8M'
    dataset_file = '/hpc/group/singhlab/rawdata/uniref50/toy_set_1000seqs.fasta'
    torch.distributed.init_process_group(backend="nccl")

    main(model_capacity=model_capacity, dataset_file=dataset_file)
    



