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

@dataclass
class RegressorTrainArgs:
    lr : int = 0.001
    num_workers :int = int(os.environ.get('OMP_NUM_THREADS'))
    epochs:int = 5
    batch_size:int = 4096

class Regressor(object):
    def __init__(self, xin_size, xout_size, datafile, train_args):
        self.xin_size = xin_size
        self.xout_size = xout_size
        self.datafile = datafile
        self.train_args =train_args

        self.model = None

    def _load_data(self):
        base_file = '/hpc/home/dgc26/projects/esm-scaling/data/train/' + self.datafile + '_'
        self.xin = np.load(base_file+self.xin_size+".npy", mmap_mode="r")
        self.xout = np.load(base_file+self.xout_size+".npy", mmap_mode="r")

        print(self.xin.shape)
        print(self.xout.shape)

    def _from_pretrained(self, file):
        self.state_dict = np.load(file, allow_pickle=True)

        regressor_hyperparams = self.state_dict["regressor_hyperparams"].item()
        valid_sgd_keys = set(SGDRegressor().get_params().keys())  

        base_estimator_params = {
            k.replace("estimator__", ""): v for k, v in regressor_hyperparams.items() if k.replace("estimator__", "") in valid_sgd_keys
        }

        self.model = MultiOutputRegressor(SGDRegressor(**base_estimator_params))
        self.model.estimators_ = []
        for coef, intercept in zip(self.state_dict["regressor_estimators"], self.state_dict["regressor_intercepts"]):
            reg = SGDRegressor(**base_estimator_params)
            reg.coef_ = np.array(coef)
            reg.intercept_ = np.array(intercept)
            self.model.estimators_.append(reg)


    def predict(self, xin, batch_size):
        n_batches = len(xin) // batch_size + 1
        y_pred = []
        for i in range(n_batches):
            start_idx = i* batch_size
            end_idx = start_idx + batch_size

            batch_X = xin[start_idx:end_idx]
            y_pred.append(self.model.predict(batch_X))

        return np.concatenate(y_pred, axis=0)


    def _predict_intern(self):
        y_pred = np.zeros_like(self.xout)
        for i in range(self.n_batches):
            start_idx = i*self.train_args.batch_size
            end_idx = start_idx + self.train_args.batch_size

            batch_X = self.xin[start_idx:end_idx]
            y_pred[start_idx:end_idx, :] = self.model.predict(batch_X)

    
        return y_pred

    def fit(self):
        base = SGDRegressor(max_iter=1, eta0=self.train_args.lr,warm_start=True)
        self.model = MultiOutputRegressor(base, n_jobs=self.train_args.num_workers)

        self._load_data()
        self.n_batches = len(self.xin) // self.train_args.batch_size + 1

        for epoch in range(self.train_args.epochs):
            for i in range(self.n_batches):
                start_idx = i*self.train_args.batch_size
                end_idx = start_idx + self.train_args.batch_size

                batch_X, batch_y = self.xin[start_idx:end_idx], self.xout[start_idx:end_idx]
                self.model.partial_fit(batch_X, batch_y)
                
            r_2 = r2_score(self.xout[:], self._predict_intern())
            print(f"At epoch {epoch}, r^2 score:{r_2}")



        self.state_dict = {
            "regressor_estimators": np.array([est.coef_ for est in self.model.estimators_], dtype=object),  
            "regressor_intercepts": np.array([est.intercept_ for est in self.model.estimators_], dtype=object), 
            "regressor_hyperparams": self.model.get_params()  
        }

        return self.state_dict

    

class PCAWrapper:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, res):
        self.mean_ = np.mean(res, axis=0)
        U, S, Vt = fbpca.pca(
            res - self.mean_,  
            k=self.n_components,
            raw=True,
            n_iter=2,  
            l=self.n_components + 10 
        )

        self.components_ = Vt
        
        self.explained_variance_ = S**2
        self.explained_variance_ratio_ =  self.explained_variance_ /  np.sum(S**2)

        self.state_dict = {
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "n_components": self.n_components
        }

        return self
    

    def transform(self, res):
        return np.dot(res - self.mean_, self.components_.T)

    def fit_transform(self, res):
        return self.fit(res).transform(res)

    def _from_pretrained(self, file):
        self.state_dict = np.load(file)

        self.mean_ = self.state_dict["mean_"]
        self.components_ = self.state_dict["components_"]
        self.explained_variance_ = self.state_dict["explained_variance_"]
        self.explained_variance_ratio_ = self.state_dict["explained_variance_ratio_"]
        self.n_components = int(self.state_dict["n_components"])




    
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
    
    

    