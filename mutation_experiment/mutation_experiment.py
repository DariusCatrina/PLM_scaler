from ..utils.util import load_mmap, models
from ..utils.config import *

from ..scaler.scaler import Scaler
from ..embeddings.embed_dataset import ESMDataset

import os
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

'''
8M experiments:
8M -> 150M

8M -> 150M -> 650M
8M -> 650M

8M -> 150M -> 650M -> 3B
8M -> 3B

150M experiments:
150M -> 650M

150M -> 650M -> 3B
150M -> 3B

650M experiments:
650M -> 3B
'''

def get_embeddings(size, dataset_name):
    base_file = "/hpc/home/dgc26/projects/esm-scaling/data/eval/"
    embed_file = f"{dataset_name}/{size}.npy"
    wilde_fle = f"{dataset_name}/{size}_wilde_type.npy"

    file_size_mut = os.path.getsize(base_file+embed_file)

    num_elements_mut = file_size_mut // np.dtype(np.float32).itemsize

    smut = np.memmap(base_file+embed_file, dtype=np.float32, mode='r', shape=(num_elements_mut// models[size]['embed_dim'],  models[size]['embed_dim']))
    s0 = np.load(base_file+wilde_fle)
    
    return s0, smut


def run_experiment_step(size_in, size_out, sin, sout, training_size):
    scaler = Scaler(xin=size_in, xout=size_out)
    scaler.from_pretrained(training_size)
    # print(f'For regressor {training_size}, {size_in}->{size_out}:')
    # print(f'intercept: {scaler.regressor.model.intercept_}')
    # print(f'coef: {scaler.regressor.model.coef_}')

    sout_prime = scaler.step(xin=sin, xout=sout)

    return sout_prime

def run_experiment_chain(size_in, size_out, seqs, training_size):
    sizes = ['8M', '150M', '650M', '3B']
    start_idx = sizes.index(size_in)
    end_idx = sizes.index(size_out)

    sin = seqs[start_idx]
    while start_idx<end_idx:
        sin = run_experiment_step(size_in=sizes[start_idx], size_out=sizes[start_idx+1], sin=sin, sout=seqs[start_idx+1], training_size=training_size)
        start_idx+=1

    return sin

def run_experiment(size_in, size_out, wilde_seqs, mut_seqs, training_size, mut_idx, is_chain):
    sizes = ['8M', '150M', '650M', '3B']
    wilde_seqs_out = wilde_seqs[sizes.index(size_out)]
    mutated_seqs_out = mut_seqs[sizes.index(size_out)]
    print(f'Chain: {is_chain}')
    if is_chain:
        wilde_seqs_prime = run_experiment_chain(size_in, size_out, seqs=wilde_seqs, training_size=training_size)
        mutated_seqs_prime = run_experiment_chain(size_in, size_out, seqs=mut_seqs, training_size=training_size)
    else:
        wilde_seqs_prime = run_experiment_step(size_in, size_out, sin=wilde_seqs[sizes.index(size_in)], sout=wilde_seqs_out,training_size=training_size)
        mutated_seqs_prime = run_experiment_step(size_in, size_out, sin=mut_seqs[sizes.index(size_in)], sout=mutated_seqs_out, training_size=training_size)


    diff = np.zeros(mutated_seqs_out.shape)
    diff_prime = np.zeros_like(diff)

    for i, idx in enumerate(mut_idx):
        diff[i] = wilde_seqs_out[idx] - mutated_seqs_out[i]
        diff_prime[i] = wilde_seqs_prime[idx] - mutated_seqs_prime[i]

    return diff, diff_prime

def run_metrics_step(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    ridge_cv = RidgeCV(alphas=(0.01,0.1,1,10, 100), cv=3).fit(X=X_train, y=y_train)
    print(f'train score: {ridge_cv.score(X_train, y_train)}, eval score: {ridge_cv.score(X_test, y_test)}, alpha: {ridge_cv.alpha_}')

    y_pred = ridge_cv.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r_2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, r_2



def run_metrics(size_in, size_out, wilde_seqs, mut_seqs, scores, mut_idx, training_size, is_chain):

    diff, diff_prime = run_experiment(size_in, size_out, wilde_seqs, mut_seqs, training_size, mut_idx, is_chain=is_chain)


    print(f'Experiment - backbone training set: {training_size}')
    mae, mse, rmse, r_2 = run_metrics_step(diff, scores)
    mae_prime, mse_prime, rmse_prime, r_2_prime = run_metrics_step(diff_prime, scores)

    print('Regression scores:')
    print(f'ESM model {size_out}:')
    print(f'MAE:{mae}\t\tMSE:{mse}\t\tRMSE{rmse}\t\tr_2 score:{r_2}')

    print(f'Scaler(Ours) {size_in} -> {size_out}')
    print(f'MAE:{mae_prime}\t\tMSE:{mse_prime}\t\tRMSE{rmse_prime}\t\tr_2 score:{r_2_prime}')
    print('\n\n\n')

    
def get_all_embeddings(dataset_name):
    wilde_seqs, mut_seqs = [], []
    sizes = ['8M', '150M', '650M', '3B']

    for size in sizes:
        wilde_seq, mut_seq = get_embeddings(size, dataset_name)
        print(f'size: {size}, shape wild: {wilde_seq.shape}, mutation: {mut_seq.shape}')
        wilde_seqs.append(wilde_seq)
        mut_seqs.append(mut_seq)

    return wilde_seqs, mut_seqs


if __name__ == '__main__':
    dataset_name = 'TADBP_HUMAN_Bolognesi_2019'
    dataset = ESMDataset(seq_file=pwd_file+'data/DMS_ProteinGym_substitutions/'+dataset_name+'.csv', tokenize=False)
    print(f'MUTATIONS ON {dataset_name}')
    wilde_seqs, mut_seqs = get_all_embeddings(dataset_name)
    mut_idx = [idx for (idx, seq) in dataset.data_dict]
    scores = dataset.scores

    print('8M experiments:')
    run_metrics('8M', '150M',wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=False)
    run_metrics('8M', '650M', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=True)
    run_metrics('8M', '650M', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=False)
    run_metrics('8M', '3B', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=True)
    run_metrics('8M', '3B', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=False)

    print('150M experiments:')
    run_metrics('150M', '650M', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=False)
    run_metrics('150M', '3B', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=False)
    run_metrics('150M', '3B', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=True)

    print('650M experiments:')
    run_metrics('650M', '3B', wilde_seqs, mut_seqs, scores, mut_idx[1:], 'toy_set_10Kseqs', is_chain=False)
    

    

    '''
8M experiments:
8M -> 150M

8M -> 150M -> 650M
8M -> 650M

8M -> 150M -> 650M -> 3B
8M -> 3B

150M experiments:
150M -> 650M

150M -> 650M -> 3B
150M -> 3B

650M experiments:
650M -> 3B
'''

