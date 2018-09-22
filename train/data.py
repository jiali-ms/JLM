import pickle
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm

sys.path.append('..')
from config import get_configs, experiment_path, data_path

class Vocab(object):
    def __init__(self, size, char_based=False):
        self.char_based = char_based
        print('vocab with size {} loaded'.format(size))
        lexicon = pickle.load(open(os.path.join(data_path, 'lexicon.pkl'), 'rb'))[:size]
        lexicon = [('<unk>', 0)] + lexicon

        if char_based:
            self.w2i = {'<unk>': 0, '<eos>': 1}
            # keep the char in the order appearance of frequent sorted word
            words = [item[0].split('/')[0] for item in lexicon if item[0] != '<eos>' and item[0] != '<unk>']
            for word in words:
                for c in word:
                    if c not in self.w2i:
                        self.w2i[c] = len(self.w2i)

            self.i2w = {v: k for k, v in self.w2i.items()}
            print('char based vocab: {} chars contained'.format(len(self.w2i)))
        else:
            self.w2i = {x[0]:i for i, x in enumerate(lexicon)}
            self.i2w = {v:k for k,v in self.w2i.items()}

        self.words = set([x[0] for i, x in enumerate(lexicon)])

    def __len__(self):
        return len(self.w2i)

class Corpus(object):
    def __init__(self, vocab, debug=False):
        self.vocab = vocab
        self.encoded_train = self.encode_corpus('train.txt', debug)
        self.encoded_dev = self.encode_corpus('dev.txt', debug)
        self.encoded_test = self.encode_corpus('test.txt', debug)

    def encode_corpus(self, filename, debug=False):
        encoded = []
        print('encode corpus: {}'.format(filename))
        with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if debug:
                lines = lines[:1024*100]
            for line in tqdm(lines):
                words = line.strip().split(' ')
                if vocab.char_based:
                    words = ''.join([word.split('/')[0] for word in words])

                encoded += [self.vocab.w2i[x] if x in self.vocab.w2i else self.vocab.w2i['<unk>'] for x in words] \
                           + [self.vocab.w2i['<eos>']]
        return encoded

if __name__ == "__main__":
    vocab = Vocab(50000, char_based=False)
    corpus = Corpus(vocab, debug=True)
    print([vocab.i2w[x] for x in corpus.encoded_train][:100])