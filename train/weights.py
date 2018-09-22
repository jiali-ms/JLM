from model import RNNLM_Model
import os
import pickle
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append('..')
from config import get_configs, experiment_path

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", "-e", type=int, default=None, help="Which experiment dump to use")
parser.add_argument("--verbose", "-v", type=bool, default=False, help="Also dump a text version of each parameter")
args = parser.parse_args()

def dump_trained_weights(experiment, verbose):
    config = get_configs(experiment)

    # Still need to load the model to build graph
    # Graph is not saved
    RNNLM_Model(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        saver.restore(session, os.path.join(experiment_path, str(experiment), "tf_dump" ,'rnnlm.weights'))

        dump_vars = ['HMi', 'HMf', 'HMo', 'HMo', 'HMg', 'IMi', 'IMf', 'IMo', 'IMg', 'LM', 'bi', 'bf', 'bo', 'bg', 'b2']

        if config.share_embedding:
            dump_vars += ['PM']
        else:
            dump_vars += ['UM']

        if config.V_table:
            dump_vars.remove('LM')
            for i, seg in enumerate(config.embedding_seg):
                if i != 0:
                    dump_vars += ['VT{}'.format(i)]
                dump_vars += ['LM{}'.format(i)]

        weight_dict = tf_weights_to_np_weights_dict(session, dump_vars)

        if config.D_softmax:
            # instead save the full patched embedding, split each block in "LM" into list of matrices
            blocks = []
            col_s = 0
            for size, s, e in config.embedding_seg:
                if e is None:
                    e = weight_dict['LM'].shape[0]
                blocks.append(weight_dict['LM'][s:e, col_s:col_s + size])
                col_s += size
            weight_dict['LM'] = blocks

        weight_dump_dir = os.path.join(experiment_path, str(experiment), "weights")
        dump_weights(weight_dict, weight_dump_dir, verbose)

def tf_weights_to_np_weights_dict(session, names):
    dict = {}
    for name in names:
        #[<tf.Variable 'Hammer/LSTM/HMi:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'Hammer/training/Hammer/LSTM/HMi/Adam:0' shape=(512, 512) dtype=float32_ref>, <tf.Variable 'Hammer/training/Hammer/LSTM/HMi/Adam_1:0' shape=(512, 512) dtype=float32_ref>]
        vars = [v for v in tf.global_variables() if name in v.name and 'training' not in v.name]
        print(vars)
        m = session.run(vars[0])
        dict[name] = m
    return dict

def dump_weights(weights_dict, dump_dir, verbose):
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    pickle.dump(weights_dict, open(os.path.join(dump_dir, "lstm_weights.pkl"), "wb"))
    print("lstm weights dict dumped in {}".format(dump_dir))

    if verbose:
        for name, m in weights_dict.items():
            if type(m) is list:
                for i, item in enumerate(m):
                    sub_name = '{}{}'.format(name, i)
                    print("dumped {} rows {}".format(sub_name, item.shape))
                    np.savetxt(os.path.join(dump_dir, sub_name + '.txt'), item)
            else:
                print("dumped {} rows {}".format(name, m.shape))
                np.savetxt(os.path.join(dump_dir, name + '.txt'), m)
                np.save(os.path.join(dump_dir, name + '.npy'), m)

        build_embedding_with_word(dump_dir)

def build_embedding_with_word(dump_dir):
    with open(os.path.join(dump_dir, "embedding.txt"), 'w') as output:
        with open(os.path.join(dump_dir, "LM.txt"), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                #print('{} {}'.format(i, line))
                output.write('{} {}'.format(i, line))

    print("embedding with word id is dumped")

'''
def build_compressed_embedding_pkl(experiment ,name):
    embeddings = []
    with open(os.path.join(experiment_path, str(experiment), "weights", name), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            v = [float(x) for x in tokens[1:]]
            embeddings.append(v)

    LM= np.array(embeddings)

    weight_dump_dir = os.path.join(experiment_path, str(experiment), "weights")
    pickle.dump(LM, open(os.path.join(weight_dump_dir, "LM.pkl"), "wb"))

    print('LM size {} dumped'.format(LM.shape))
'''

if __name__ == "__main__":
    dump_trained_weights(args.experiment, args.verbose)