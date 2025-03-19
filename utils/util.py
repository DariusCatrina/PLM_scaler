import esm
from config import *
import os
import numpy as np

def read_fasta_to_dict(fasta_file):
    fasta_dict = []
    with open(fasta_file, 'r') as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id:
                    fasta_dict.append((sequence_id,''.join(sequence)))
                
                sequence_id = line[1:].split()[0]
                sequence = []  
            else:
               
                sequence.append(line)
        
       
        if sequence_id:
            fasta_dict.append((sequence_id,''.join(sequence)))

    return fasta_dict


def _download_model_alphabet_wrap(model_cap):
    
    model_name = models[model_cap]['name']
    
    file = f'/hpc/group/singhlab/rawdata/esm_pretrained_models/{model_name}.pt'
    model, alphabet = esm.pretrained.load_model_and_alphabet(file)        

    return model, alphabet

def get_model_tuple(model_capacity):
    return *_download_model_alphabet_wrap(model_capacity), models[model_capacity]['layers']


def load_mmap(size, dataset, split):
    base_file = f'/hpc/home/dgc26/projects/esm-scaling/data/{split}/{dataset}/'
    file_size = os.path.getsize(base_file+size+".npy")

    num_elements = file_size // np.dtype(np.float32).itemsize
    size = size.split('_')[0]
    x = np.memmap(base_file+size+".npy", dtype=np.float32, mode='r', shape=(num_elements// models[size]['embed_dim'],  models[size]['embed_dim']))

    return x