import h5py
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from multiprocessing import Pool
from sklearn.metrics import r2_score
from dataclasses import dataclass
import argparse

import os
import numpy as np
import joblib
import pickle
import fbpca

from ..utils.config import *
from .modules import *

@dataclass
class RegressorTrainArgs:
    lr : int = 0.001
    num_workers :int = int(os.environ.get('OMP_NUM_THREADS'))
    epochs:int = 5
    batch_size:int = 4096


class Scaler(object):
    def __init__(self, xin, xout):
        self.xin = xin
        self.xout = xout
        self.regressor = Regressor(xin, xout, None, None)
        self.pca = PCAWrapper(n_components=models[self.xout]['embed_dim'] - models[self.xin]['embed_dim'])

        self.state_dict = {}

    def fit_regressor(self, datafile, train_args):
        print('Fitting regresor..')
        self.regressor.datafile = datafile
        self.regressor.train_args = train_args

        regressor_state_dict = self.regressor.fit()
        self.state_dict.update(regressor_state_dict)

    def fit_pca(self, residuals):
        print('Fitting pca..')
        self.pca.fit(residuals)
        pca_state_dict = self.pca.state_dict
        self.state_dict.update(pca_state_dict)

    def transform_pca(self, residuals):
        return self.pca.transform(residuals)
    
    def from_pretrained(self, datafile):
        base_file = '/hpc/home/dgc26/projects/esm-scaling/data/'
        file = base_file+f"scaler_{self.xin}_{self.xout}_{datafile}.npz"
        self.regressor._from_pretrained(file)
        self.pca._from_pretrained(file)

    def predict_regressor(self, xin, batch_size):
        return self.regressor.predict(xin, batch_size)

    def fit(self, datafile, train_args, verbose=True):
        self.fit_regressor(datafile, train_args)

        xin_transformed = self.predict_regressor(self.regressor.xin, train_args.batch_size)
        res = self.regressor.xout - xin_transformed
        self.fit_pca(res)
        
        if verbose:
            print(f'fitted pca:{self.transform_pca(res)}')
        self.save_state_dict()

    def step(self, xin, xout, batch_size):
        xin_transformed = self.predict_regressor(xin, batch_size)
        res = xout - xin

        reduced_xin = self.transform_pca(res)
        xout_prime = np.concatenate((xin, reduced_xin), axis=-1)

        return xout_prime
    

    def save_state_dict(self):
        print(f'Saving Scaler model: {self.xin}->{self.xout}')
        base_file = '/hpc/home/dgc26/projects/esm-scaling/data/'
        np.savez_compressed(base_file+ f"scaler_{self.xin}_{self.xout}_{self.regressor.datafile}.npz", **self.state_dict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scaler training script")
    parser.add_argument("--model_capacity_in", type=str, default='8M')
    parser.add_argument("--model_capacity_out", type=str, default='150M')
    parser.add_argument("--dataset_file", type=str, default='None', help="Protein file")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None)


    args = parser.parse_args()

    xin = args.model_capacity_in
    xout = args.model_capacity_out
    training_args = RegressorTrainArgs(epochs=args.epochs, batch_size=args.batch_size)
    
    datafile = args.dataset_file
    print(f'{xin} -> {xout}')
    scaler = Scaler(xin=xin, xout=xout)
    scaler.fit(datafile=datafile, train_args=training_args)
