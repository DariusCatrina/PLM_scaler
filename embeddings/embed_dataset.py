import math
import numpy as np
from pandas import read_csv

from util import *

import torch
from torch.utils.data import Dataset, DistributedSampler


class ESMDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle=False

        total_size = len(self.dataset)
        self.split_sizes = [(total_size // num_replicas) + (1 if i < total_size % num_replicas else 0)
                            for i in range(num_replicas)]

        self.offsets = [sum(self.split_sizes[:i]) for i in range(num_replicas)]
        self.local_size = self.split_sizes[self.rank]

    def __iter__(self):
        start = self.offsets[self.rank]
        end = start + self.local_size

        return iter(list(range(start, end)))

    def __len__(self):
        return self.local_size

def get_mut_idx(mutation_str):
     # XiY (Y is in the ith position, X originally in ith position)
    i = int(mutation_str[1:-1]) - 1

    return i

class ESMDataset(Dataset):
    def __init__(self, batch_converter=None, seq_file=None, tokenize=True):
        self.batch_converter = batch_converter
        self.tokens_lens = None
        self.tokens_labels = None
        self.tokens = None
        self.scores = None

        if seq_file.split('.')[-1] == 'fasta':
            self.init_data_fasta(seq_file)
        if seq_file.split('.')[-1] == 'csv':
            self.init_data_mut(seq_file)

        if tokenize:
            self.tokenize()


        self.dataset_name = seq_file.split('/')[-1].split('.')[0]

    def tokenize(self):
        self.tokens_lens = np.zeros((len(self.data_dict)), dtype=np.int16)
        for i, (_, seq) in enumerate(self.data_dict):
            self.tokens_lens[i] = len(seq)
        
        self.tokens_labels, _, self.tokens = self.batch_converter(self.data_dict)

    def init_data_fasta(self, seq_file):

        self.data_dict = read_fasta_to_dict(seq_file)


    def init_data_mut(self, seq_file):
        data_pd = read_csv(seq_file) 
        # reconstruct original protein
        first_mut = data_pd['mutant'][0] 
        X = first_mut[0]
        i = get_mut_idx(first_mut)

        original_prot = list(data_pd['mutated_sequence'][0])
        original_prot[i] = X
        original_prot = ''.join(original_prot)

        mutated_prot_arr = data_pd['mutated_sequence']
        mutated_idx_arr = data_pd['mutant'].apply(get_mut_idx)

        data = list(zip(mutated_idx_arr, mutated_prot_arr))
        data = [(-1, original_prot)] + data

        self.data_dict = data
        self.scores = data_pd['DMS_score'].to_numpy()
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return idx, self.tokens_lens[idx], self.tokens_labels[idx], self.tokens[idx]


