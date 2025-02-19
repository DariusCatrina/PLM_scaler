import torch
from torch.utils.data import Dataset
from util import read_fasta_to_dict
import numpy as np

class ESMDataset(Dataset):
    def __init__(self, batch_converter=None, seq_file=None):
        self.batch_converter = batch_converter
        self.tokens_lens = None
        self.tokens_labels = None
        self.tokens = None

        if seq_file.split('.')[-1] == 'fasta':
            self.init_data_fasta(seq_file)

        self.dataset_name = seq_file.split('/')[-1].split('.')[0]

    def init_data_fasta(self, seq_file):
        data_dict = read_fasta_to_dict(seq_file)
        self.tokens_lens = np.zeros((len(data_dict)), dtype=np.int32)

        for i, (_, seq) in enumerate(data_dict):
            self.tokens_lens[i] = len(seq)
        self.tokens_labels, _, self.tokens = self.batch_converter(data_dict)


    def init_data_mut(self, seq_file):
        raise NotImplementedError()

    def __len__(self):
        assert len(self.tokens_lens) == len(self.tokens_lens) == len(self.tokens)

        return len(self.tokens)

    def __getitem__(self, idx):
        return idx, self.tokens_lens[idx], self.tokens_labels[idx], self.tokens[idx]


