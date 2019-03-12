import pickle
import numpy as np
import os
import json
from random import shuffle
import sys
from decoder import Decoder, CharRNNDecoder
from decoder_dynamic import DynamicDecoder
from decoder_ngram import NGramDecoder
sys.path.append('..')
from config import data_path, experiment_path
from tqdm import tqdm
from train.data import Vocab, CharVocab
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_id", "-e", type=int, default=1, help="experiment id to eval")
parser.add_argument("--eval_size", "-es", type=int, default=100, help="Number of sentences to evaluate")
parser.add_argument("--use_ngram", "-ng", type=bool, default=False, help="Use ngram decoder or not")
parser.add_argument("--ngram_order", "-o", type=int, default=3, help="Ngram order")
parser.add_argument("--comp", "-c", type=int, default=0, help="Compression bit, 0 means no compression")
parser.add_argument("--vocab_select", "-vs", type=bool, default=False, help="Use vocab select method or not")
parser.add_argument("--top_sampling", "-ts", type=bool, default=False, help="Sampling strategy for vocab select")
parser.add_argument("--random_sampling", "-rs", type=bool, default=False, help="Sampling strategy for vocab select")
parser.add_argument("--samples", "-s", type=int, default=0, help="Samples when using advanced sampling")
parser.add_argument("--beam_size", "-b", type=int, default=10, help="Beam size for decoder")
parser.add_argument("--dynamic_decoding", "-dd", type=bool, default=False, help="Use incremental decoding or not")

args = parser.parse_args()

class Evaluator:
    def __init__(self):
        self.config = json.loads(open(os.path.join(experiment_path, str(args.experiment_id), 'config.json'), 'rt').read())
        if self.config['char_rnn']:
            self.vocab = CharVocab(self.config['vocab_size'])
        else:
            self.vocab = Vocab(self.config['vocab_size'])
        self.w2i = self.vocab.w2i

        if args.use_ngram:
            self.decoder = NGramDecoder(experiment_id=args.experiment_id, ngram_order=args.ngram_order)
        elif self.config['char_rnn']:
            self.decoder = CharRNNDecoder(experiment_id=args.experiment_id, comp=args.comp)
        elif args.dynamic_decoding:
            self.decoder = DynamicDecoder(experiment_id=args.experiment_id, comp=args.comp)
        else:
            self.decoder = Decoder(experiment_id=args.experiment_id, comp=args.comp)

    def evaluate(self):
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

        with open('eval/eval_log_{}_e_{}_dynamic_{}_size_{}_b_{}_comp_{}_vocab_sel_{}_samples_{}_top_{}_random_{}.txt'.format(
                decoder_type,
                args.experiment_id,
                args.dynamic_decoding,
                args.eval_size,
                args.beam_size,
                args.comp,
                args.vocab_select,
                args.samples,
                args.top_sampling,
                args.random_sampling
        ), 'w', encoding='utf-8') as f:

            x_, y_ = self.load_eval_set()

            start_time = time.time()

            for x, y in tqdm(zip(x_, y_), total=args.eval_size):
                results = self.decoder.decode(x, beam_width=args.beam_size, vocab_select=args.vocab_select, samples=args.samples,
                                              top_sampling=args.top_sampling, random_sampling=args.random_sampling)
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

            f.write('best_hit {} nbest_hit{} no_hit {} eval_size {}'.format(best_hit, n_best_hit,
                                                                            args.eval_size-best_hit-n_best_hit,
                                                                            args.eval_size))

            if not args.use_ngram:
                f.write("--- %f seconds lstm per step ---" % (np.mean(self.decoder.perf_log_lstm)))
                f.write("--- %f seconds softmax per step ---" % (np.mean(self.decoder.perf_log_softmax)))
                f.write("--- %f seconds per sent.---" % (np.sum(self.decoder.perf_log_lstm + self.decoder.perf_log_softmax) / self.decoder.perf_sen))

            f.write("--- %s seconds ---" % (time.time() - start_time))

            print('best_hit {} nbest_hit{} no_hit {} eval_size {}'.format(best_hit, n_best_hit,
                                                                          args.eval_size-best_hit-n_best_hit,
                                                                          args.eval_size))
            if not args.use_ngram:
                print("--- %f seconds lstm per step ---" % (np.mean(self.decoder.perf_log_lstm)))
                print("--- %f seconds softmax per step ---" % (np.mean(self.decoder.perf_log_softmax)))
                print("--- %f seconds per sent.---" % (np.sum(self.decoder.perf_log_lstm + self.decoder.perf_log_softmax) / self.decoder.perf_sen))

            if args.dynamic_decoding:
                print("--- %f seconds per step for vocab fix.---" % np.mean(self.decoder.perf_log_fix_vocab))
                print("--- %f seconds per step for lattice path fix.---" % np.mean(self.decoder.perf_log_fix_lattice_path_prob))
            print("--- %f seconds ---" % (time.time() - start_time))


    def load_eval_set(self, debug=True):
        """
        Read the test portion of the corpus.

        :return: Reading and sentence pairs. The sentences that contains oov will be removed.
        """
        def has_oov(tokens):
            for token in tokens:
                if self.decoder._check_oov(token):
                    return True
            return False

        x = []
        y = []


        with open(os.path.join(data_path, 'test.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print('take {} for evaluation from all {} lines'.format(args.eval_size, len(lines)))

            use_short_sentences = False
            if use_short_sentences:
                print('-------------sentence selection used')

            for line in lines:
                tokens = line.strip().split(' ')
                if not has_oov(tokens):
                    readings = ''.join([x.split('/')[1] if x.split('/')[1] != '' else x.split('/')[0] for x in tokens])
                    target = ''.join([x.split('/')[0] for x in tokens])

                    if use_short_sentences:
                        if len(readings) > 30:
                            continue

                    x.append(readings)
                    y.append(target)

                    if len(x) >= args.eval_size:
                        break

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


if __name__ == '__main__':
    eval = Evaluator()
    eval.evaluate()