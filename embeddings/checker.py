import os
import numpy as np
from util import *
from config import *
import pickle as pkl
import time
from embed_dataset import *
from torch.utils.data import DataLoader

def load_mmap(size):
    start_time = time.time()
    base_file = '/hpc/home/dgc26/projects/esm-scaling/data/train/toy_set_1000seqs_'
    file_size = os.path.getsize(base_file+size+"_multiGPU.npy")

    num_elements = file_size // np.dtype(np.float32).itemsize

    x = np.memmap(base_file+size+"_multiGPU.npy", dtype=np.float32, mode='r', shape=(num_elements// models[size]['embed_dim'],  models[size]['embed_dim']))

    print(f'Loading time for {size} - mmap:{time.time() - start_time}')
    return x


def flatten_tuple(tuple_list):
    
    x = tuple_list[0][1]
    for i in range(1, len(tuple_list)):
        x = np.concatenate((x, tuple_list[i][1]))
        
    return x


def load_old(size):
    start_time = time.time()
    base_file = '/hpc/home/dgc26/projects/esm-scaling/data/train/'
    data_file = size + '__toy_set_1000seqs_notebook'

    x = pkl.load(open(base_file+data_file, 'rb'))
    x = flatten_tuple(x)

    print(f'Loading time for {size} - notebook:{time.time() - start_time}')

    return x

def init_data_fasta(seq_file):
    data_dict = read_fasta_to_dict(seq_file)
    total_len = 0

    for i, (_, seq) in enumerate(data_dict):
        total_len+= len(seq)
    
    print(total_len)
    #FINAL SHAPE: (28069, 320)
    # 26185, 26185
    # 552550
    # 5492589

if __name__ == '__main__':
    init_data_fasta('/hpc/group/singhlab/rawdata/uniref50/toy_set.fasta')
    # init_data_fasta('/hpc/group/singhlab/rawdata/uniref50/toy_set_1000seqs.fasta')
    # init_data_fasta('/hpc/group/singhlab/rawdata/uniref50/toy_set_10Kseqs.fasta')


    # model, alphabet, repr_layer = get_model_tuple('8M')
    # batch_converter = alphabet.get_batch_converter()
    # embeddings_size = models['8M']['embed_dim']

    # dataset = ESMDataset(batch_converter=batch_converter, seq_file='/hpc/group/singhlab/rawdata/uniref50/toy_set.fasta')
    # dataloader = DataLoader(dataset, batch_size=34, num_workers=int(os.getenv('OMP_NUM_THREADS')), pin_memory=True)

    # prot_len = 0
    # for (_, batch_len, _,_) in dataloader:
    #     prot_len+=sum(batch_len)

    # print(prot_len)
        

   