import h5py
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from multiprocessing import Pool
from sklearn.metrics import r2_score
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor

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
        self.train_args = train_args

        self.model = None
        self.dtype=np.float32
        self.scaler = MinMaxScaler()
        self.scale = False

        self.embeddings_in = models[self.xin_size]['embed_dim']
        self.embeddings_out = models[self.xout_size]['embed_dim']


    def _load_data(self):
        base_file = f'/hpc/home/dgc26/projects/esm-scaling/data/train/{self.datafile}/'
        data_file_in = base_file +self.xin_size+".npy"
        data_file_out = base_file +self.xout_size+".npy"

        file_size_in = os.path.getsize(data_file_in)
        file_size_out = os.path.getsize(data_file_out)
        num_elements_in = file_size_in // np.dtype(self.dtype).itemsize
        num_elements_out = file_size_out // np.dtype(self.dtype).itemsize
        
        self.xin = np.memmap(data_file_in, dtype=np.float32, mode='c', shape=(num_elements_in // self.embeddings_in, self.embeddings_in))
        self.xout = np.memmap(data_file_out, dtype=np.float32, mode='c', shape=(num_elements_out // self.embeddings_out, self.embeddings_out))

        if self.scale:
            n_batches = len(self.xin) // self.train_args.batch_size + 1
            print('Training scaler...')
            print(self.xin)
            for i in range(n_batches):
                start_idx = i*self.train_args.batch_size
                end_idx = start_idx + self.train_args.batch_size
                batch_X = self.xin[start_idx:end_idx]
                self.scaler.partial_fit(batch_X)

            print('Scaling seq in...')
            for i in range(n_batches):
                start_idx = i*self.train_args.batch_size
                end_idx = start_idx + self.train_args.batch_size
                self.xin[start_idx:end_idx] = self.scaler.transform(self.xin[start_idx:end_idx])
            # print(self.xin)
        print(self.xin.shape, self.xout.shape)

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
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_X = xin[start_idx:end_idx]

            y_pred.append(self.model.predict(batch_X))

        return np.concatenate(y_pred, axis=0)


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
  
        r_2 = r2_score(self.xout, self.predict(self.xin, self.train_args.batch_size))
        print(f"r^2 score:{r_2}")



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


    