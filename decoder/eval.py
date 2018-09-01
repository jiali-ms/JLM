import pickle
import numpy as np
import os
import json
from random import shuffle
import sys
from decoder import Decoder
from decoder_ngram import NGramDecoder
sys.path.append('..')
from config import data_path, experiment_id, experiment_path
from tqdm import tqdm
from train.data import Vocab
import time

class Evaluator:
    def __init__(self, use_ngram=False):
        if use_ngram:
            self.decoder = NGramDecoder()
        else:
            self.decoder = Decoder()

        self.config = json.loads(open(os.path.join(experiment_path, str(experiment_id), 'config.json'), 'rt').read())
        vocab = Vocab(self.config['vocab_size'])
        self.w2i = vocab.w2i

    def evaluate(self, samples=200):
        """
        Run decoding with decoder with a fixed number of pairs from evaluation set.

        :param samples:
        :return:
        """
        best_hit = 0
        n_best_hit = 0

        if isinstance(self.decoder, NGramDecoder):
            decoder_type =  "ngram"
        else:
            decoder_type = "neural"

        with open('eval_log_e{}_{}.txt'.format(experiment_id, decoder_type), 'w', encoding='utf-8') as f:
            x_, y_ = self.load_eval_set()

            start_time = time.time()

            for x, y in tqdm(zip(x_[:samples], y_[:samples]), total=samples):
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
            f.write("--- %s seconds ---" % (time.time() - start_time))

            print('best_hit {} nbest_hit{} no_hit {} samples {}'.format(best_hit, n_best_hit, samples-best_hit-n_best_hit, samples))
            print("--- %s seconds ---" % (time.time() - start_time))

    def load_eval_set(self, debug=True):
        """
        Read the test portion of the corpus.

        :return: Reading and sentence pairs. The sentences that contains oov will be removed.
        """
        def has_oov(tokens):
            for token in tokens:
                if token not in self.w2i:
                    return True
            return False

        x = []
        y = []


        with open(os.path.join(data_path, 'test.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print('{} lines'.format(len(lines)))
            for line in lines:
                tokens = line.strip().split(' ')
                if not has_oov(tokens):
                    readings = ''.join([x.split('/')[1] if x.split('/')[1] != '' else x.split('/')[0] for x in tokens])
                    target = ''.join([x.split('/')[0] for x in tokens])
                    x.append(readings)
                    y.append(target)

            print('{} pairs load'.format(len(x)))
            return x, y

if __name__ == '__main__':
    eval = Evaluator(use_ngram=False)
    eval.evaluate(samples=2000)