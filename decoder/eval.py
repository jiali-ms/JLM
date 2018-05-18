import pickle
import numpy as np
import os
import json
from random import shuffle
import sys
from decoder import Decoder
sys.path.append('..')
from config import data_path, experiment_id
from tqdm import tqdm

class Evaluator:
    def __init__(self):
        self.decoder = Decoder()

    def evaluate(self, samples=200):
        """
        Run decoding with decoder with a fixed number of pairs from evaluation set.

        :param samples:
        :return:
        """
        best_hit = 0
        n_best_hit = 0

        with open('eval_log_e{}.txt'.format(experiment_id), 'w', encoding='utf-8') as f:
            x_, y_ = self.load_eval_set()
            for x, y in tqdm(zip(x_[:samples], y_[:samples])):
                results = self.decoder.decode(x)
                # convert to list of strings
                sentences = [''.join([x.split('/')[0] for x in item[1]]) for item in results]
                if y == sentences[0]:
                    best_hit += 1
                    f.write('best hit\n')
                elif y in sentences:
                    f.write('nbest hit\n')
                    n_best_hit += 1
                else:
                    f.write('no hit\n')

                f.write('{}\t{}\n'.format(y, x))
                for item in sentences:
                    f.write('{}\n'.format(item))

            f.write('best_hit {} nbest_hit{} no_hit {} samples {}'.format(best_hit, n_best_hit, samples-best_hit-n_best_hit, samples))
            print('best_hit {} nbest_hit{} no_hit {} samples {}'.format(best_hit, n_best_hit, samples-best_hit-n_best_hit, samples))

    def load_eval_set(self, debug=True):
        """
        Read the test portion of the corpus.

        :return: Reading and sentence pairs. The sentences that contains oov will be removed.
        """
        def has_oov(tokens):
            for token in tokens:
                if '<unk_' in token:
                    return True
            return False

        x = []
        y = []

        i2w = pickle.load(open(os.path.join(data_path, 'i2w.pkl'), 'rb'))
        encoded_corpus = pickle.load(open(os.path.join(data_path, 'encoded_corpus.pkl'), 'rb'))
        size = len(encoded_corpus)
        test = encoded_corpus[round(size * 0.9): size]
        if debug:
            test = test[:10000]

        print(len(test))

        tokens = []
        for id in test:
            token = i2w[id]
            if token == '<eos>':
                # start of current sentence
                if len(tokens) > 0 and not has_oov(tokens):
                    # print(tokens)
                    readings = [x.split('/')[1] if x.split('/')[1] != '' else x.split('/')[0] for x in tokens]
                    words = [x.split('/')[0] for x in tokens]
                    x.append(''.join(readings))
                    y.append(''.join(words))

                tokens = []

            else:
                tokens.append(token)

        print('{} pairs load'.format(len(x)))

        return x, y

if __name__ == '__main__':
    eval = Evaluator()
    eval.evaluate()