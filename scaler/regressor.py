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

    def _from_pretrained(self):
        base_file = '/hpc/home/dgc26/projects/esm-scaling/data/'
        self.state_dict = joblib.load(base_file+f"scaler_"+self.xin_size+'_'+self.xout_size+".pkl")
        
        valid_sgd_keys = set(SGDRegressor().get_params().keys())  
        base_estimator_params = {
            k.replace("estimator__", ""): v for k, v in self.state_dict["regressor_hyperparams"].items() if k.replace("estimator__", "") in valid_sgd_keys
        }
        base_estimator = SGDRegressor(**base_estimator_params)  
        self.model = MultiOutputRegressor(base_estimator)
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
            "regressor_estimators": [est.coef_ for est in self.model.estimators_],  # List of coef arrays
            "regressor_intercepts": [est.intercept_ for est in self.model.estimators_],  # List of intercepts
            "regressor_hyperparams": self.model.get_params()  # Save model hyperparameters
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
        self.mean_ = np.mean(X, axis=0)
        U, S, Vt = fbpca.pca(
            X - self.mean_,  
            k=self.n_components,
            raw=True,
            n_iter=2,  
            l=self.n_components + 10 
        )

        self.components_ = Vt
        
        self.explained_variance_ = S**2
        self.explained_variance_ratio_ =  self.explained_variance_ /  np.sum(S**2)

        return self

    def transform(self, res):
        return np.dot(res - self.mean_, self.components_.T)

    def fit_transform(self, res):
        return self.fit(res).transform(res)


    
class Scaler(object):
    def __init__(self, xin, xout):
        self.xin = xin
        self.xout = xout
        self.regressor = Regressor(xin, xout, None, None)

        self.state_dict = {}

    def train_regressor(self, datafile, train_args):
        self.regressor.datafile = datafile
        self.regressor.train_args = train_args

        regressor_state_dict = self.regressor.fit()
        self.state_dict.update(regressor_state_dict)
    
    def _from_pretrained(self):
        self.regressor._from_pretrained()

    def predict_regressor(self, xin, batch_size):
        return self.regressor.predict(xin, batch_size)

    def save_state_dict(self):
        print(f'Saving Scaler model: {self.xin}->{self.xout}')
        base_file = '/hpc/home/dgc26/projects/esm-scaling/data/'
        joblib.dump(self.state_dict,base_file+ f"scaler_{self.xin}_{self.xout}.pkl", compress=3)



if __name__ == '__main__':
    regressor_trainargs = RegressorTrainArgs(epochs=7)
    scaler = Scaler(xin='8M', xout='150M')
    scaler.train_regressor(datafile='toy_set_1000seqs', train_args=regressor_trainargs)
    scaler.save_state_dict()

    print('Loading state dict')
    scaler._from_pretrained()

    xin = np.load('/hpc/home/dgc26/projects/esm-scaling/data/train/toy_set_1000seqs_8M.npy', mmap_mode="r")
    xout = np.load('/hpc/home/dgc26/projects/esm-scaling/data/train/toy_set_1000seqs_150M.npy', mmap_mode="r")

    print('Running inferance using Scaler 8M->150M:')
    y_pred = scaler.predict_regressor(xin, regressor_trainargs.batch_size)
    print(xout.shape, y_pred.shape)
    print(f'R2 score: {r2_score(xout[:],y_pred)}')

    res = xout - y_pred
    pca = PCAWrapper(n_components=models['150M']['embed_dim'] - models['8M']['embed_dim'])

    res_pca = pca.fit_transform(res)
    print('Scaler pca:')
    print(res_pca)

    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA 

    def flatten_tuple(tuple_list):
        x = tuple_list[0][1]
        for i in range(1, len(tuple_list)):
            x = np.concatenate((x, tuple_list[i][1]))
            
        return x
    xin = flatten_tuple(pickle.load(open(f'./data/train/8M__toy_set_1000seqs', 'rb')))
    xout =  flatten_tuple(pickle.load(open(f'./data/train/150M__toy_set_1000seqs', 'rb')))
    reg1 = LinearRegression()
    reg1.fit(X=xin, y=xout)
    y_pred = reg1.predict(xin)
    print(f'R2 score sklearn regression: {r2_score(xout, y_pred)}')

    res = xout - y_pred
    pca = PCA(n_components=models['150M']['embed_dim'] - models['8M']['embed_dim'])
    res_pca = pca.fit_transform(res)
    print('Scaler pca:')
    print(res_pca)


    