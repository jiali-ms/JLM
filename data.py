from collections import defaultdict
import os
import numpy as np
import sys
import operator
import pickle
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", "-v", type=int, default=100000, help="Vocab size to build a corpus with")
args = parser.parse_args()

def build_lexicon(corpus_path):
    """Build lexicon from a prepared corpus.
    Make sure the corpus is segmented by space already and each line is a sentence.

    Output:
        lexicon.pkl: a sorted list with all the words and their frequency of appearance.

    A simple analysis of the corpus is also printed out to understand the coverage of corpus by top n words.
    """
    lexicon = defaultdict(int)
    with open(corpus_path, 'r', encoding='utf-8') as f:
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
    # Note sort with value in dict cannot guarantee same order each time for items with value.
    sorted_words = sorted(lexicon.items(), key=operator.itemgetter(1), reverse=True)
    print('most frequently words are:')
    print(sorted_words[:100])

    print('tokens distribution:')
    dist = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    for d in dist:
        p = np.sum([f for w, f in sorted_words[:d]]) / len(sorted_words)
        print('top {} words: {}'.format(d, round(p, 2)))

    # dump lexicon
    pickle.dump(sorted_words, open('data/lexicon.pkl', 'wb'))
    print('lexicon dumped')

    _build_reading_dict()

def _build_reading_dict():
    """For Japanese or Chinese decoder, reading dictionary is a mapping of homonym - words with same pronunciations.
    In this implementation, a word is in the format of display/reading/POS where each reading is the key for the dict.

    Output:
        reading_dict.pkl: key is reading, value is a list of indices of words in lexicon.
                    !! Note that the index in reading_dict is the index of words in lexicon.pkl !!
    """
    reading_dict = defaultdict(list)
    lexicon = pickle.load(open('data/lexicon.pkl', 'rb'))
    # use the index of a word in the list
    for i, word in enumerate(lexicon):
        tokens = word[0].split('/')
        if len(tokens) < 3:
            # <eos>, <unk>
            continue
        display = tokens[0]
        reading = tokens[1]
        if reading == '':
            reading = display
        reading_dict[reading].append(i)
    print('reading dict dumped with {} keys'.format(len(reading_dict.keys())))
    sorted_reading = sorted(reading_dict.items(), key=lambda x: len(x[1]), reverse=True)
    print('most frequently shared readings')
    for reading, l in sorted_reading[:20]:
        print('{}: {}'.format(reading, len(l)))
    pickle.dump(reading_dict, open(os.path.join('data/reading_dict.pkl'), 'wb'))
    print('reading dict dumped')

def build_training_corpus(vocab_size):
    """
    Mark the oov in the original corpus with <unk_pos>.
    And put the new training corpus in a new folder named corpus_{vocab_size}.
    :param vocab_size: top n words to be included

    Output:
        corpus.txt: a corpus for training with unk words marked.
        w2i (word to index): a dict with word as key, index as key.
        i2w (index to word): reverse dict of word to index.
        u2f (unk to frequency): frequency of <unk_pos>, used for oov
    """
    dir = 'data/corpus_{}'.format(vocab_size)
    if not os.path.exists(dir):
        os.makedirs(dir)

    lexicon = pickle.load(open('data/lexicon.pkl', 'rb'))
    vocab = defaultdict(int, {x[0]: x[1] for x in lexicon[:vocab_size]})
    with open('{}/corpus.txt'.format(dir), 'w', encoding='utf-8') as corpus:
        with open('data/corpus.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i % 1000 == 0:
                    sys.stdout.write('\rprogress: {}%'.format(round(i/len(lines)*100, 2)))
                    sys.stdout.flush()
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

    # Also dump a copy of w2i, i2w for the training corpus and u2f for decoder
    # Note that the index is ordered by word's frequency. This is vital for algorithm like D-softmax.
    sorted_words = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    w2i = {x[0]: i for i, x in enumerate(sorted_words)}
    i2w = {value: key for key, value in w2i.items()}
    u2f = {w: f for w, f in sorted_words if '<unk_' in w}
    pickle.dump(u2f, open('{}/u2f.pkl'.format(dir), 'wb'))
    pickle.dump(w2i, open('{}/w2i.pkl'.format(dir), 'wb'))
    pickle.dump(i2w, open('{}/i2w.pkl'.format(dir), 'wb'))
    print('training corpus dumped')

    _build_encoded_corpus(w2i, dir)

def _build_encoded_corpus(w2i, dir, shuffle=True):
    """Build a corpus with word turned to index and dump to a pickle for easy loading in training.

    Output:
        encoded_corpus.pkl: all the words and sentences connected into one list
    """
    encoded_corpus = []
    with open('{}/corpus.txt'.format(dir), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % 1000 == 0:
                sys.stdout.write('\rprogress: {}%'.format(round(i/len(lines)*100, 2)))
                sys.stdout.flush()
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
    print('encoded corpus dumped')

if __name__ == "__main__":
    if not os.path.exists('data/lexicon.pkl'):
        build_lexicon('data/corpus.txt')
    build_training_corpus(args.vocab_size)


