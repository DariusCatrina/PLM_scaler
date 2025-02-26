from ..embeddings.embed_dataset import ESMDataset
from ..utils.util import *
from ..utils.config import *


model_capacity = '8M'

model, alphabet, repr_layer = get_model_tuple(model_capacity)
batch_converter = alphabet.get_batch_converter()

dataset = ESMDataset(batch_converter, '/hpc/home/dgc26/projects/esm-scaling/data/DMS_ProteinGym_substitutions/BRCA1_HUMAN_Findlay_2018.csv')
print(dataset[0])
print(dataset[1])
print(dataset[2])
print(dataset[3])





