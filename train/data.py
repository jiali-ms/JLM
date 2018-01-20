import pickle
import numpy as np
import os
import sys
from collections import defaultdict
sys.path.append('..')
from config import experiment_path, experiment_id

'''
Pack the compressed embeddings into a pickle again  
'''
def build_compressed_embedding_pkl(name):
    embeddings = []
    weights_dir = os.path.join(experiment_path, str(experiment_id), "weights")
    with open(os.path.join(weights_dir, name), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            v = [float(x) for x in tokens[1:]]
            embeddings.append(v)

    LM= np.array(embeddings)
    pickle.dump(LM, open(os.path.join(weights_dir, "CLM.pkl"), "wb"))

    print('LM size {} dumped'.format(LM.shape))

if __name__ == "__main__":
    # test the model
    build_compressed_embedding_pkl('embedding.txt.comp')
