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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_ngram", "-ng", type=bool, default=False, help="Use ngram decoder or not")
parser.add_argument("--ngram_order", "-o", type=int, default=3, help="Ngram order")
parser.add_argument("--samples", "-s", type=int, default=2000, help="Number of sentences to evaluate")
parser.add_argument("--comp", "-c", type=int, default=0, help="Compression bit, 0 means no compression")

args = parser.parse_args()

class Evaluator:
    def __init__(self):
        if args.use_ngram:
            self.decoder = NGramDecoder(ngram_order=args.ngram_order)
        else:
            self.decoder = Decoder(args.comp)

        self.config = json.loads(open(os.path.join(experiment_path, str(experiment_id), 'config.json'), 'rt').read())
        vocab = Vocab(self.config['vocab_size'])
        self.w2i = vocab.w2i

    def evaluate(self, samples=args.samples):
        """
        Run decoding with decoder with a fixed number of pairs from evaluation set.

        :param samples:
        :return:
        """
        best_hit = 0
        n_best_hit = 0

        if isinstance(self.decoder, NGramDecoder):
            decoder_type = "ngram_{}".format(args.ngram_order)
        else:
            decoder_type = "neural"

        with open('eval_log_e{}_{}_comp_{}.txt'.format(experiment_id, decoder_type, args.comp), 'w', encoding='utf-8') as f:

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

def parse_log():
    for folder, subs, files in os.walk('./'):
            for filename in files:
                if 'eval_log' in filename:
                    print(filename)
                    path = os.path.join(folder, filename)
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            if 'best_hit' in line:
                                print(line.strip())

parse_log()

if __name__ == '__main__':
    eval = Evaluator()
    eval.evaluate()