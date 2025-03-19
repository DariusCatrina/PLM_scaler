import numpy as np
import pickle
import os

from ..utils.config import *
from ..utils.util import *

from ..scaler.scaler import Scaler


def flatten_tuple(tuple_list):
    
    x = tuple_list[0][1]
    for i in range(1, len(tuple_list)):
        x = np.concatenate((x, tuple_list[i][1]))
        
    return x

def load_old(size, dataset_name, split):
    base_file = f"/hpc/home/dgc26/projects/esm-scaling/data/{split}/"
    embed_file = f"{size}__{dataset_name}"

    tuple_list = pickle.load(open(base_file+embed_file, 'rb'))
    s0 = [tuple_list[0]]
    smut = flatten_tuple(tuple_list[1:])

    return s0, smut

def load_new(size, dataset_name):
    base_file = "/hpc/home/dgc26/projects/esm-scaling/data/eval/"
    embed_file = f"{dataset_name}/{size}.npy"
    wilde_fle = f"{dataset_name}/{size}_wilde_type.npy"

    s0 = np.load(base_file+wilde_fle)

    file_size_mut = os.path.getsize(base_file+embed_file)


    num_elements_mut = file_size_mut // np.dtype(np.float32).itemsize

    smut = np.memmap(base_file+embed_file, dtype=np.float32, mode='r', shape=(num_elements_mut// models[size]['embed_dim'],  models[size]['embed_dim']))
    
    return s0, smut

def get_small(dataset_name, size):
    base_file = f"/hpc/home/dgc26/projects/esm-scaling/data/eval/"
    wilde_fle = f"{dataset_name}/{size}_wilde_type.npy"

    return np.load(base_file+wilde_fle)

if __name__ == "__main__":
    dataset = 'BRCA1_HUMAN_Findlay_2018'

    training_size = 'toy_set'
    scaler = Scaler(xin='8M', xout='650M')
    scaler.from_pretrained(training_size)
 
    print(scaler.regressor.model.intercept_)
    print(scaler.regressor.model.coef_)

    
    training_size = 'toy_set_10Kseqs'
    scaler = Scaler(xin='8M', xout='650M')
    scaler.from_pretrained(training_size)

    print(scaler.regressor.model.intercept_)
    print(scaler.regressor.model.coef_)
    # print(load_old('150M', dataset_name))
    # print(load_new('150M', dataset_name))

    # print(load_old('650M', dataset_name))
    # print(load_new('650M', dataset_name))

