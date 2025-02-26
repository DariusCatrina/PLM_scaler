from modules import *
from config import *
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA 
from sklearn.metrics import r2_score
import pickle as pkl

def flatten_tuple(tuple_list):
    
    x = tuple_list[0][1]
    for i in range(1, len(tuple_list)):
        x = np.concatenate((x, tuple_list[i][1]))
        
    return x


def load_old(size, dataset):
    base_file = '/hpc/home/dgc26/projects/esm-scaling/data/train/'
    data_file = size + f'__{dataset}_notebook'

    x = pkl.load(open(base_file+data_file, 'rb'))
    x = flatten_tuple(x)

    print(x.shape)

    return x


def load_mmap(size, dataset):
    base_file = f'/hpc/home/dgc26/projects/esm-scaling/data/train/{dataset}/'
    file_size = os.path.getsize(base_file+size+".npy")

    num_elements = file_size // np.dtype(np.float32).itemsize

    x = np.memmap(base_file+size+".npy", dtype=np.float32, mode='r', shape=(num_elements// models[size]['embed_dim'],  models[size]['embed_dim']))
    print(x.shape)
    return x



def lin_reg_old(size_in, size_out, dataset):
    x_in, x_out = load_old(size_in, dataset), load_old(size_out, dataset)
    print(f'{size_in} -> {size_out}')
    print(x_in.shape, x_out.shape)

    reg = LinearRegression(n_jobs=int(os.getenv('OMP_NUM_THREADS')))
    reg.fit(X=x_in, y=x_out)
    x_outprime = reg.predict(x_in)

    print(f'r^2 score (notebook): {r2_score(x_out, x_outprime)}')

    pca_1 = PCA(n_components=models[size_out]['embed_dim'] - models[size_in]['embed_dim'])
    res = x_out - x_outprime
    print(pca_1.fit_transform(res))


def lin_reg_new(size_in, size_out, dataset):
    training_args = RegressorTrainArgs(epochs=10)
    reg = Regressor(size_in, size_out, dataset, training_args)
    xin, xout = load_mmap(size_in, dataset), load_mmap(size_out, dataset)
    reg.fit()
    
    xout_prime = reg.predict(xin, training_args.batch_size)

    print(f'R2 score (multiGPU): {r2_score(xout, xout_prime)}')

def load_shape(size, dataset):
    datafile = f'/hpc/home/dgc26/projects/esm-scaling/data/train/{dataset}/{size}.npy'
    sizes= np.load(datafile)

    print(sizes)
if __name__ == '__main__':
    dataset = 'toy_set'
    size_in='8M'
    size_out='150M'
    # lin_reg_old('8M', '150M', dataset)
    # lin_reg_old('150M', '650M', dataset)
    # lin_reg_old('650M', '3B', dataset)
    from modules import PCAWrapper

    x_in= load_old(size_in, dataset)
    pca_1 = PCA(n_components=models[size_out]['embed_dim'] - models[size_in]['embed_dim'])
    print(pca_1.fit_transform(x_in))

    pca_2 = PCAWrapper(n_components=models[size_out]['embed_dim'] - models[size_in]['embed_dim'])
    pca_2.fit(x_in)

    print(pca_2.transform(x_in))



