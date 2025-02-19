from embed_dataset import ESMDataset
from util import *
from config import *
import h5py
import os
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import gc


def run_embedding_generation(rank, model, world_size, dataset, model_args):
    
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model = DDP(model, device_ids=[rank])
    model.eval()

    repr_layer, batch_size, embeddings_size = model_args['repr_layer'], model_args['batch_size'], model_args['embeddings_size']
    model_capacity = model_args["model_capacity"]

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    
    if rank == 0:
        print(f"Running embedding generation on {world_size} GPUs...")

    print('********* Sanity check *********\n')
    print(f'On device {rank}')
    print(f'Length of dataset: {len(dataset)}, batch_size: {batch_size}')
    print(f'Dataloader length: {len(dataloader)}')
    print('*********')

    rank_embeddings = []
    rank_len = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            _, batch_lens, _, batch_tokens = batch
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer] # B x L x E

            max_len = token_representations.size(dim=1)
            curr_batch_len = torch.sum(batch_lens).item()

            selected_embeddings = [token_representations[i, 1: 1 + batch_lens[i], :] for i in range(len(batch_lens))]
            flattened_embeddings = torch.cat(selected_embeddings, dim=0).detach().to('cpu').numpy()
            
            rank_embeddings.append(flattened_embeddings)
            rank_len+=flattened_embeddings.shape[0]
            

    dist.barrier()

    gathered_results = [None] * world_size 
    dist.all_gather_object(gathered_results, rank_len)

    print(f'**** Saving on {rank} ****\n')
    base_file = '/hpc/home/dgc26/projects/esm-scaling/data/train/'
    embed_file = f'{dataset.dataset_name}_{model_capacity}_{rank}.npy'

    np.save(base_file+embed_file, np.concatenate(rank_embeddings, axis=0))
    
    
    if rank == 0:
        print('******* Generation complete *****\n')
        print('Merging....')
        final_file =  f'{dataset.dataset_name}_{model_capacity}.npy'
        mmap_final = np.memmap(base_file+final_file, dtype=np.float32, mode='w+', shape=(sum(gathered_results), embeddings_size))
        offset = 0

        for rank_ in range(world_size):
            embed_file = f"{dataset.dataset_name}_{model_capacity}_{rank_}.npy"
            local_embed = np.load(base_file+embed_file)
            local_len = local_embed.shape[0]

            mmap_final[offset:offset + local_len] = local_embed
            mmap_final.flush()
            del local_embed
            os.remove(base_file+embed_file)
            gc.collect()
            offset+=local_len
            
    dist.barrier()

def main(model_capacity, dataset_file, batch_size):
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)

    
    if dist.get_rank() == 0:
        print(f'Loading the {model_capacity} model...')
    model, alphabet, repr_layer = get_model_tuple(model_capacity)
    batch_converter = alphabet.get_batch_converter()
    embeddings_size = models[model_capacity]['embed_dim']

    if dist.get_rank() == 0:
        print('Loading the dataset...')
    dataset = ESMDataset(batch_converter=batch_converter, seq_file=dataset_file)

    model_args = {}
    model_args['model_capacity'], model_args['repr_layer'], model_args['batch_size'], model_args['embeddings_size'] = model_capacity, repr_layer, batch_size, embeddings_size

    run_embedding_generation(rank, model, world_size, dataset, model_args)
    dist.destroy_process_group()


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Embedding generation script using PLMs.")
    parser.add_argument("--model_capacity", type=str, default='8M', help="The model used for generating")
    parser.add_argument("--dataset_file", type=str, default='toy_set_1000seqs.fasta', help="Protein file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    model_capacity = args.model_capacity

    basefile = '/hpc/group/singhlab/rawdata/uniref50/'
    dataset_file = basefile + args.dataset_file
    batch_size = args.batch_size

    main(model_capacity=model_capacity, dataset_file=dataset_file, batch_size=batch_size)
    


