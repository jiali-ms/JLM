from collections import defaultdict
import os
import numpy as np
import sys
import operator
import pickle
import random

'''
Make sure the corpus is a list of sentences where words are separated by space.

Output:
lexicon: a list with all the words and their frequency of appearance.

A simple analysis of the corpus is also printed out to understand the coverage of corpus by top n words.  
'''
def build_lexicon(path):
    lexicon = defaultdict(int)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            words = line.strip('\n').split(' ')
            if len(words) == 0:
                continue
            lexicon['<eos>'] += 1  # one per sentence in corpus
            for word in words:
                lexicon[word] += 1

    # print out lexicon analysis
    lexicon_size = len(lexicon.items())
    print('{} words in corpus'.format(lexicon_size))
    sorted_words = sorted(lexicon.items(), key=operator.itemgetter(1), reverse=True)
    print('most frequently words are:')
    print(sorted_words[:100])

    print('tokens distribution:')
    num_tokens = np.sum([x for w, x in sorted_words])
    for i in range(100):
        x = np.sum([x for w, x in sorted_words[:round(lexicon_size * (i+1) / 100)]])
        print('top {}%: {}'.format(i+1, x / num_tokens))

    print('top 50k words: {}'.format(np.sum([x for w, x in sorted_words[:50000]]) / num_tokens))
    print('top 100k words: {}'.format(np.sum([x for w, x in sorted_words[:100000]]) / num_tokens))

    _dump_lexicon(lexicon, 'data')


'''
With lexicon build, decide the vocabulary size and filter out the oov by mark them '<unk_pos>'
'''
def build_corpus_with_limited_vocab(top_n = 50000):
    dir = 'data/corpus_{}'.format(top_n)
    if not os.path.exists(dir):
        os.makedirs(dir)

    lexicon = pickle.load(open('data/lexicon.pkl', 'rb'))
    vocab = defaultdict(int, {x[0]: x[1] for x in lexicon[:top_n]})

    with open('{}/corpus.txt'.format(dir), 'w', encoding='utf-8') as corpus:
        with open('data/corpus.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    print(i / len(lines))
                new_line = []
                words = line.strip('\n').split(' ')
                for word in words:
                    if word in vocab.keys():
                        new_line.append(word)
                    else:
                        pos = word.split('/')[-1]
                        unk = '<unk_{}>'.format(pos)
                        new_line.append(unk)
                        vocab[unk] += 1

                corpus.write('{}\n'.format(' '.join(new_line)))

    _dump_lexicon(vocab, dir)


'''
w2i (word to index): a dict with word as key, index as key. Note that the index is ordered by word's frequency. 
i2w (index to word): reverse dict of word to index.
'''
def _dump_lexicon(lexicon, dir):
    # Note that the python sort with dict items can not ensure same result for the words with same frequency
    # Dump it to make sure the later on pipeline has reliable vocab
    sorted_words = sorted(lexicon.items(), key=operator.itemgetter(1), reverse=True)
    w2i = {x[0]: i for i, x in enumerate(sorted_words)}
    i2w = {value: key for key, value in w2i.items()}
    pickle.dump(sorted_words, open('{}/lexicon.pkl'.format(dir), 'wb'))
    pickle.dump(w2i, open('{}/w2i.pkl'.format(dir), 'wb'))
    pickle.dump(i2w, open('{}/i2w.pkl'.format(dir), 'wb'))


'''
Build a pickle that has all the words encoded and connected.
'''
def build_encoded_corpus(dir, shuffle=True):
    w2i = pickle.load(open('{}/w2i.pkl'.format(dir), 'rb'))
    encoded_corpus = []
    with open('{}/corpus.txt'.format(dir), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % 10000 == 0:
                print(i / len(lines))
            words = line.strip('\n').split(' ')
            encoded_words = [w2i[w] for w in words]
            encoded_corpus.append(encoded_words)

    if shuffle:
        random.shuffle(encoded_corpus)

    result = []
    for line in encoded_corpus:
        result.extend(line)
        result.append(w2i['<eos>'])

    pickle.dump(result, open('{}/encoded_corpus.pkl'.format(dir), 'wb'))

#build_lexicon('data/corpus.txt')
#build_corpus_with_limited_vocab(50000)
#build_corpus_with_limited_vocab(100000)
build_encoded_corpus('data/corpus_50000')
