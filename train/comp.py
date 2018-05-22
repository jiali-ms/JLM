import numpy as np
import pickle
import time
import argparse
import os
import sys
sys.path.append('..')
from sklearn.cluster import KMeans

from config import experiment_path

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", "-e", type=int, default=None, help="Which experiment dump to use")
args = parser.parse_args()

# https://arxiv.org/pdf/1510.00149v5.pdf
# deep compression use k-means for quantization

def kmeans_compress(weight, bit=8):
    """
    compress weights using k-means

    It is a method mentioned in the paper deep compression. The input will be the original raw weights.
    Output will be compressed code like 0-255 and its centroids as its codebook.

    It takes about 1 hour to train a embedding in size like (50k, 512) with 8 CPUs cores all running.

    :param weight:
    :param bit:
    :return: code, centroids(codebook)
    """

    shape = weight.shape
    weight = weight.reshape(-1, 1)

    assert bit <= 32
    clusters = 2 ** bit
    print('{} clusters'.format(clusters))
    kmeans = KMeans(n_clusters=clusters, n_jobs=4)
    kmeans.fit(weight)
    code = kmeans.predict(weight)

    if bit <= 8:
        code = code.astype(np.uint8)

    centroids = kmeans.cluster_centers_
    return code.reshape(shape), centroids.astype('f')

def compressed_trained_weights(experiment, debug=True):
    # make the comp dir
    weights_path = os.path.join(experiment_path, str(experiment), "weights")
    if not os.path.exists(os.path.join(weights_path, 'comp')):
        os.makedirs(os.path.join(weights_path, 'comp'))

    weights = pickle.load(open(os.path.join(weights_path, 'lstm_weights.pkl'), 'rb'))

    # compress each weight use kmeans
    comp_weights = {}
    comp_dump = {}
    for key, value in weights.items():
        print('compressing {} {} '.format(key, value.shape))
        start = time.time()
        code, codebook = kmeans_compress(value, bit)

        end = time.time()
        comp_dump[key] = (code, codebook)
        comp_weights[key] = np.take(codebook, code)

        if debug:
            np.savetxt(os.path.join(weights_path, 'comp', '{}_code.txt'.format(key)), code.astype(int), fmt='%i')
            np.savetxt(os.path.join(weights_path, 'comp', '{}_codebook.txt'.format(key)), codebook)

        print('{} {} compressed in {} s'.format(key, value.shape, end-start))

    # dump the compressed (code, codebook), also keep a decoded version
    pickle.dump(comp_weights, open(os.path.join(weights_path, 'lstm_weights_comp.pkl'), 'wb'))
    pickle.dump(comp_dump, open(os.path.join(weights_path, 'comp', 'lstm_weights_comp_dump.pkl'), 'wb'))

if __name__ == '__main__':
    if args.experiment is not None:
        compressed_trained_weights(args.experiment)
    else:
        compressed_trained_weights("10")

    '''
    comp_weights = {}
    comp_dump = {}
    weights_path = os.path.join(experiment_path, str(9), "weights")
    names = ['HMi', 'HMf', 'HMo', 'HMg', 'IMi', 'IMf', 'IMo', 'IMg', 'bi', 'bf', 'bo', 'bg', 'LM', 'b2', 'PM']
    for name in names:
        code = np.loadtxt(os.path.join(weights_path, 'comp', '{}_code.txt'.format(name)), dtype=np.dtype('B'))
        codebook = np.loadtxt(os.path.join(weights_path, 'comp', '{}_codebook.txt'.format(name)), dtype=np.dtype('f'))
        comp_dump[name] = (code, codebook)
        comp_weights[name] = np.take(codebook, code)

    pickle.dump(comp_weights, open(os.path.join(weights_path, 'lstm_weights_comp.pkl'), 'wb'))
    pickle.dump(comp_dump, open(os.path.join(weights_path, 'comp', 'lstm_weights_comp_dump.pkl'), 'wb'))
    '''
