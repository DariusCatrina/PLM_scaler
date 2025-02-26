from ..embeddings.embed_dataset import ESMDataset, ESMDistributedSampler
from ..embeddings.generate import main as gen_main
from ..utils.util import *
from ..utils.config import *
import h5py
import os
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import gc


def run_embedding_generation(rank, model, world_size, dataset, model_args):
    
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model = DDP(model, device_ids=[rank])
    model.eval()

    repr_layer, batch_size, embeddings_size = model_args['repr_layer'], model_args['batch_size'], model_args['embeddings_size']
    model_capacity = model_args["model_capacity"]

    sampler = ESMDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=int(os.getenv('OMP_NUM_THREADS')), pin_memory=True)

    
    if rank == 0:
        print(f"Running embedding generation on {world_size} GPUs...")

    print('********* Sanity check *********\n')
    print(f'On device {rank}')
    print(f'Length of dataset: {len(dataset)}, batch_size: {batch_size}')
    print(f'Dataloader length: {len(dataloader)}')
    print('*********')

    rank_embeddings = []
    rank_idx = []
    wilde_type = None
    rank_len = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            idx, batch_lens, batch_labels, batch_tokens = batch
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer] # B x L x E

            selected_embeddings = []
            for i in range(len(batch_tokens)):
                mut_idx = batch_labels[i]

                if mut_idx == None:
                    wilde_type = token_representations[i, 1: 1 + batch_lens[i], :].detach().to('cpu').numpy()
                    assert wilde_type.size() == torch.Size([batch_lens[i], embeddings_size])
                else:
                    selected_embeddings.append(token_representations[i, 1 + mut_idx, :])
                    rank_idx.extend(idx)
                    assert selected_embeddings[-1].size() == torch.Size([1, embeddings_size])

            selected_embeddings = selected_embeddings.detach().to('cpu').numpy()
            if wilde_type == None:
                assert selected_embeddings.shape == (len(idx), embeddings_size)
            else:
                assert selected_embeddings.shape == (len(idx) - 1, embeddings_size)
            rank_embeddings.extend(selected_embeddings)
            rank_len+= selected_embeddings.shape[0]


    dist.barrier()

    gathered_results = [None] * world_size 
    dist.all_gather_object(gathered_results, rank_len)
    print(f'**** Saving on {rank} ****\n')
    base_file = f'/hpc/home/dgc26/projects/esm-scaling/data/eval/{dataset.dataset_name}/'
    embed_file = f'{model_capacity}_{rank}.npy'
    idx_file = f'{model_capacity}_{rank}_idx.npy'

    if wilde_type != None:
        wild_type_file = f'{model_capacity}_wilde_type.npy'
         np.save(base_file+wild_type_file, wilde_type)
    
    np.save(base_file+embed_file, np.concatenate(rank_embeddings, axis=0))
    np.save(base_file+idx_file, np.array(rank_idx))
    
    if rank == 0:
        print('******* Generation complete *****\n')
        print('Merging....')
        final_file =  f'{model_capacity}.npy'

        print(f'FINAL SHAPE: {sum(gathered_results), embeddings_size}')
        mmap_final = np.memmap(base_file+final_file, dtype=np.float32, mode='w+', shape=(sum(gathered_results), embeddings_size))
        offset = 0

        for rank_ in range(world_size):
            embed_file = f"{model_capacity}_{rank_}.npy"
            idx_file = f'{model_capacity}_{rank_}_idx.npy'
            local_embed = np.load(base_file+embed_file)
            local_idx = np.load(base_file+idx_file)
            
            assert local_idx.shape[0] == local_embed.shape[0]
            
            for i,idx in enumerate(local_idx):
                mmap_final[idx, :] = local_embed[i]
                
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
    parser.add_argument("--dataset_file", type=str, default='None', help="Protein file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    model_capacity = args.model_capacity

    basefile = '/hpc/home/dgc26/projects/esm-scaling/data/DMS_ProteinGym_substitutions/'
    dataset_file = basefile + args.dataset_file
    batch_size = args.batch_size

    main(model_capacity=model_capacity, dataset_file=dataset_file, batch_size=batch_size)
    


