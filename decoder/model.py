import pickle
import numpy as np
import os
import json
from random import shuffle
import sys
sys.path.append('..')
from config import data_path, experiment_path
from train.data import Vocab
import time

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def softmax(w):
    assert w.ndim == 2 or w.ndim == 1, 'softmax dim error %d' % w.ndim
    w = np.expand_dims(w, axis=0) if w.ndim == 1 else w
    e = np.exp(w - np.amax(w, axis=1, keepdims=True))
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist

def tanh(x):
    return np.tanh(x)

def find_top_N(a,N):
    return np.argsort(a)[::-1][:N]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

class LSTM_Model():
    """The numpy implementation of the NN Language Model"""
    def __init__(self, experiment_id=0, comp=0):
        print('LSTM model: exp {} comp {}'.format(experiment_id, comp))
        self.config = json.loads(open(os.path.join(experiment_path, str(experiment_id), "config.json"), "rt").read())
        self.weights = self._load_model(experiment_id, comp)
        self.embed_size = self.config['embed_size']
        self.hidden_size = self.config['hidden_size']
        self.share_embedding = self.config['share_embedding']
        self.hidden = np.zeros((1, self.hidden_size))
        self.cell = np.zeros((1, self.hidden_size))

        if self.config['D_softmax']:
            self.blocks = self.weights['LM']
            self.embed_size = sum([x[0] for x in self.config['embedding_seg']])
            self.weights['LM'] = np.zeros((self.weights['b2'].shape[0], self.embed_size))
            col_s = 0
            for i, (size, s, e) in enumerate(self.config['embedding_seg']):
                self.weights['LM'][s:e, col_s:col_s + size] = self.blocks[i]
                col_s += size

        if self.config['V_table']:
            self.blocks = []
            self.v_tables = []
            embeddings = []
            for i, seg in enumerate(self.config['embedding_seg']):
                block = self.weights['LM{}'.format(i)]
                self.blocks.append(block)
                if i != 0:
                    v_table = self.weights['VT{}'.format(i)]
                    self.v_tables.append(v_table)
                    embeddings.append(np.dot(block, v_table))
                else:
                    self.v_tables.append(None)
                    embeddings.append(block)

            self.weights['LM'] = np.concatenate(embeddings, axis=0)

    def _load_model(self, experiment_id=0, comp=0):
        if comp:
            file = 'lstm_weights_comp_{}.pkl'.format(comp)
            print('use compressed model, comp_{}'.format(comp))
        else:
            file = 'lstm_weights.pkl'

        weights = pickle.load(open(os.path.join(experiment_path, str(experiment_id), 'weights', file), 'rb'))

        # temp code for shu compression algorithm
        # make this a model parameter once done
        use_hash_code = False
        if use_hash_code:
            M = 32
            K = 16
            code = np.loadtxt(open(os.path.join(experiment_path, str(experiment_id), 'weights', 'LM.codes'), 'r'), dtype=np.int)
            codebook = np.load(open(os.path.join(experiment_path, str(experiment_id), 'weights', 'LM.codebook.npy'), 'rb'))

            LM = np.zeros((code.shape[0], codebook[0].shape[0]))
            for row in range(LM.shape[0]):
                c = code[row]
                LM[row, :] = np.sum([codebook[i*K+x] for i, x in enumerate(c)], axis=0)

            print('hash coded embedding loaded')

            print('distance {}'.format(np.mean(np.linalg.norm(LM - weights['LM'], axis=1).tolist())))

            weights['LM'] = LM

            np.save('LM.npy', LM)

        return weights

    def predict(self, index, vocab=None, reset=False, self_norm=False):
        if reset: # hidden and cell should be set before using this function
            self.hidden = np.zeros(shape=self.hidden.shape)
            self.cell = np.zeros(shape=self.cell.shape)

        start_time = time.time()
        self._lstm_cell(index)
        log_lstm_time = time.time() - start_time

        start_time = time.time()
        y = self.project(self.hidden, vocab)
        if not self_norm:
            pred = softmax(y)
        log_softmax_time = time.time() - start_time

        return pred, y, log_lstm_time, log_softmax_time

    def _lstm_cell(self, index):
        # embedding lookup
        e = self.weights['LM'][index,:]
        i = np.dot(self.hidden, self.weights['HMi']) + np.dot(e, self.weights['IMi']) + self.weights['bi']
        f = np.dot(self.hidden, self.weights['HMf']) + np.dot(e, self.weights['IMf']) + self.weights['bf']
        o = np.dot(self.hidden, self.weights['HMo']) + np.dot(e, self.weights['IMo']) + self.weights['bo']
        g = np.dot(self.hidden, self.weights['HMg']) + np.dot(e, self.weights['IMg']) + self.weights['bg']
        i = sigmoid(i)
        f = sigmoid(f)
        o = sigmoid(o)
        g = tanh(g)

        self.cell = np.multiply(self.cell, f) + np.multiply(g, i)
        # new hidden transformation matrix
        self.hidden = np.multiply(tanh(self.cell), o)

    def project(self, hidden, vocab=None):
        # output word representation
        if self.share_embedding:
            if self.config['D_softmax']:
                temp = np.dot(hidden, self.weights["PM"])
                y = []
                col_s = 0
                for i, (size, s, e) in enumerate(self.config['embedding_seg']):
                    if vocab:
                        if e is None:
                            e = sys.maxsize
                        sub_vocab_lookup = [v - s for v in vocab if v >= s and v < e]
                        y.append(np.dot(temp[:, col_s: col_s + size], self.blocks[i][sub_vocab_lookup].T))
                    else:
                        y.append(np.dot(temp[:, col_s: col_s + size], self.blocks[i].T))
                    col_s += size
                if vocab:
                    y = np.concatenate(y, axis=1) + self.weights['b2'][vocab]
                else:
                    y = np.concatenate(y, axis=1) + self.weights['b2']
            elif self.config['V_table']:
                temp = np.dot(hidden, self.weights["PM"])
                y = []
                for i, (size, s, e) in enumerate(self.config['embedding_seg']):
                    if vocab:
                        if e is None:
                            e = sys.maxsize
                        sub_vocab_lookup = [v - s for v in vocab if v >= s and v < e]
                        if i != 0:
                            y.append(np.dot(np.dot(temp, self.v_tables[i].T), self.blocks[i][sub_vocab_lookup].T))
                        else:
                            y.append(np.dot(temp, self.blocks[i][sub_vocab_lookup].T))
                    else:
                        if i != 0:
                            y.append(np.dot(np.dot(temp, self.v_tables[i].T), self.blocks[i].T))
                        else:
                            y.append(np.dot(temp, self.blocks[i].T))
                if vocab:
                    y = np.concatenate(y, axis=1) + self.weights['b2'][vocab]
                else:
                    y = np.concatenate(y, axis=1) + self.weights['b2']
            else:
                if vocab:
                    y = np.dot(np.dot(hidden, self.weights["PM"]), self.weights["LM"][vocab].T) + self.weights['b2'][vocab]
                else:
                    y = np.dot(np.dot(hidden, self.weights["PM"]), self.weights["LM"].T) + self.weights['b2']
        else:
            if vocab:
                y = np.dot(hidden, self.weights['UM'][vocab]) + self.weights['b2'][vocab]
            else:
                y = np.dot(hidden, self.weights['UM']) + self.weights['b2']

        return y

    def predict_with_context(self, index, hidden, cell, vocab=None):
        self.hidden = hidden
        self.cell = cell
        return self.predict(index, vocab), self.hidden, self.cell

    def evaluate(self, start, inputs):
        probs = []
        pred = self.predict([start], vocab=None, reset=True)
        for input in inputs:
            probs.append(pred[0, input])
            pred = self.predict([input])
        return [-np.log(p) for p in probs]

def show_prob(inputs):
    results = model.evaluate(w2i['<eos>'], [w2i[word] for word in inputs])
    print(results)
    print(sum([np.log(prob) for prob in results]))

if __name__ == "__main__":
    # test the model
    experiment_id = 22
    config = json.loads(open(os.path.join(experiment_path, str(experiment_id), "config.json"), "rt").read())
    vocab = Vocab(config['vocab_size'], char_based=config['char_rnn'])
    i2w = vocab.i2w
    w2i = vocab.w2i
    model = LSTM_Model(experiment_id=experiment_id)
    starting_text = '<eos>'
    result = []
    step = 0

    #print(sum(model.evaluate(w2i[starting_text], [w2i[x] for x in 'ことも無'])))
    #print(sum(model.evaluate(w2i[starting_text], [w2i[x] for x in 'こともむ'])))

    while True:
        result.append(starting_text)
        pred = model.predict([w2i[starting_text]])
        pred = pred[0].tolist()
        next_idx = sample(pred)
        starting_text = i2w[next_idx]
        step = step + 1
        if step == 100:
            break

    print('--- generated sentence')
    print(' '.join([x.split('/')[0] for x in result]))
    show_prob(result)

    print('--- random sentence by same collection of words, check the difference to see if the model is correct')
    shuffle(result)
    print(' '.join([x.split('/')[0] for x in result]))
    show_prob(result)