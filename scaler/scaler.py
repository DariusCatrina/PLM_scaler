import h5py
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from multiprocessing import Pool
from sklearn.metrics import r2_score
from dataclasses import dataclass

import os
import numpy as np
import joblib
import pickle
import fbpca

from config import *
from modules import *

class Scaler(object):
    def __init__(self, xin, xout):
        self.xin = xin
        self.xout = xout
        self.regressor = Regressor(xin, xout, None, None)
        self.pca = PCAWrapper(n_components=models['150M']['embed_dim'] - models['8M']['embed_dim'])

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
    
    def from_pretrained(self):
        base_file = '/hpc/home/dgc26/projects/esm-scaling/data/'
        file = base_file+f"scaler_"+self.xin+'_'+self.xout+".npz"
        self.regressor._from_pretrained(file)
        self.pca._from_pretrained(file)

    def predict_regressor(self, xin, batch_size):
        return self.regressor.predict(xin, batch_size)
    
    def transform_pca(self, residuals):
        return self.pca.transform(residuals)

    def fit(self, datafile, train_args):
        self.fit_regressor(datafile, train_args)

        xin_transformed = self.predict_regressor(self.regressor.xin, train_args.batch_size)
        res = self.regressor.xout - xin_transformed
        self.fit_pca(res)

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
        np.savez_compressed(base_file+ f"scaler_{self.xin}_{self.xout}.npz", **self.state_dict)


if __name__ == '__main__':
        
    scaler = Scaler(xin='8M', xout='150M')
    scaler.fit(datafile='toy_set_1000seqs', train_args= RegressorTrainArgs(epochs=13))

    scaler = Scaler(xin='150M', xout='650M')
    scaler.fit(datafile='toy_set_1000seqs', train_args=RegressorTrainArgs(epochs=7))

    scaler = Scaler(xin='650M', xout='3B')
    scaler.fit(datafile='toy_set_1000seqs', train_args=RegressorTrainArgs(epochs=2))