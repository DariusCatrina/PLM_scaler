# PLM Scaler 

Content of the repo
```
PLM_Scaller
    > embeddings
          |''' directory that contains the code for embedding generation (protein sequence to ESM embedding) '''
          |--> generate.py (runs the pipeline)
          |--> emebed_datasets.py (dataset class where each item is a tuple formed of (protein length, protein name, tokenized  protein))
          |--> util.py (usefull auxiliary functions)
          |--> config.py (configuration file)
    > scaler 
        | ''' directory that contains the code for Regression + PCA pipeline '''
        |TODO
    > run.sh (sh file to submit to slurm)
```