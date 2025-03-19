base_file = '/hpc/group/singhlab/rawdata/uniref50/'
file_10k = 'toy_set_10Kseqs'
file_toy = 'toy_set' 
pwd_file = '/hpc/home/dgc26/projects/esm-scaling/'

models = {
    '8M' : {"name": "esm2_t6_8M_UR50D",'layers':6, 'embed_dim':320},
    '150M' : {"name": "esm2_t30_150M_UR50D", "layers": 30, "embed_dim":640},
    '650M': {"name": "esm2_t33_650M_UR50D", 'layers':33, 'embed_dim':1280},
    '3B':{"name": "esm2_t36_3B_UR50D", 'layers':36, 'embed_dim':2560}
}
