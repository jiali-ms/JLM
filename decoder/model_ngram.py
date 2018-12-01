import pickle
import numpy as np
import os
import json
from random import shuffle
import math
import sys
sys.path.append('..')
from config import data_path, experiment_path

class NGramModel():
    """N-gram model that take arpa file as input and provide probability with previous words.

        Arpa parsing code are mainly from
        https://raw.githubusercontent.com/yohokuno/neural_ime/master/decode_ngram.py
    """
    def __init__(self, ngram_file='lm3', ngram_order=3):
        self.ngram_order = ngram_order
        self.model = self.parse_srilm(os.path.join(data_path, ngram_file))

    def parse_ngram(self, ngram):
        for word in ngram.split(' '):
            if word == '<s>':
                yield '<eos>'
            elif word == '</s>':
                yield '<eos>'
            else:
                yield word

    def parse_srilm(self, file):
        print('{} loaded'.format(file))
        ngrams = {}
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                fields = line.split('\t', 2)

                if len(fields) < 2:
                    continue

                if len(fields) == 2:
                    logprob, ngram = fields
                    backoff = None
                elif len(fields) > 2:
                    logprob, ngram, backoff = fields
                    backoff = -math.log(10 ** float(backoff))
                cost = -math.log(10 ** float(logprob))
                ngram = tuple(self.parse_ngram(ngram))
                ngrams[ngram] = (cost, backoff)

        print('{} ngrams loaded'.format(len(ngrams)))
        return ngrams

    def predict(self, words, debug=False):
        if type(words) is list:
            words = tuple(words[-self.ngram_order:])
        if words in self.model:
            cost, _ = self.model[words]
            if debug:
                print(words)
            return cost

        if len(words) == 1:
            return 100.0

        return self.predict(words[1:], debug)

    def evaluate(self, words, debug=False):
        prob = []
        words = ['<eos>'] + words
        for i in range(2, len(words)+1):
            prob.append(self.predict(words[:i], debug))
            if debug:
                print(prob[-1])
        return sum(prob)

if __name__ == "__main__":
    # test the model
    model = NGramModel(ngram_file='lm3', ngram_order=2)
    print(model.evaluate(['今日/キョー/名詞-普通名詞-副詞可能', 'は/ワ/助詞-係助詞', 'いい/イー/形容詞-非自立可能', '天気/テンキ/名詞-普通名詞-一般', 'です/デス/助動詞'], debug=True))
