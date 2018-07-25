from collections import defaultdict
import os
import numpy as np
import pickle
from random import shuffle
from tqdm import tqdm

def build_lexicon(corpus_path):
    """Build lexicon from a prepared corpus.

    Make sure the corpus is segmented by space already and each line is a sentence.
    A simple analysis of the corpus is also printed out to understand the coverage of corpus by top n words.

    Output:
        lexicon.pkl: a sorted list with all the words and their frequency of appearance.
        lexicon.txt: raw text dump
    """

    # parse corpus file and build word count
    lexicon = defaultdict(int)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            words = line.strip('\n').split(' ')
            if len(words) == 0:
                continue
            lexicon['<eos>'] += 1  # one per sentence in corpus
            for word in words:
                lexicon[word] += 1

    # print out lexicon analysis
    # Sort frequency then alphabetically to ensure order
    sorted_words = sorted(lexicon.items(), key=lambda x: (-x[1], x[0]))
    total_token = sum([x[1] for x in sorted_words])
    print('{} tokens in corpus with vocab {}'.format(total_token, len(sorted_words)))

    print('tokens distribution:')
    dist = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    for d in dist:
        p = np.sum([f for w, f in sorted_words[:d]]) / total_token
        print('top {} words: {}'.format(d, round(p, 2)))

    # dump lexicon pickle
    pickle.dump(sorted_words, open('data/lexicon.pkl', 'wb'))

    # dump lexicon raw text
    with open('data/lexicon.txt', 'w', encoding='utf-8') as f:
        for k, v in sorted_words:
            f.write('{}\t{}\n'.format(k, v))

    print('lexicon dumped')

    build_reading_dict(sorted_words)

    return sorted_words

def build_reading_dict(lexicon):
    """Build reading to word mapping.

    Reading dictionary is used to build lattice from user input. For Japanese, the reading is kana.
    For Chinese, the reading is pinyin.

    Output:
        reading_dict.pkl: key is reading, value is a list of indices of words in lexicon.
    """
    reading_dict = defaultdict(list)
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

    # print metadata
    print('reading dict dumped with {} keys'.format(len(reading_dict.keys())))
    sorted_reading = sorted(reading_dict.items(), key=lambda x: len(x[1]), reverse=True)
    print('most frequently shared readings')
    for reading, l in sorted_reading[:20]:
        print('{}: {}'.format(reading, len(l)))

    # dump reading dict as pickle
    pickle.dump(reading_dict, open(os.path.join('data/reading_dict.pkl'), 'wb'))

    # dump reading dict as raw text
    with open('data/reading_dict.txt', 'w', encoding='utf-8') as f:
        for x in sorted_reading:
            words = [lexicon[idx][0] for idx in x[1]]
            f.write('{}\t{}\n'.format(x[0], ' '.join(words)))

    print('reading dict dumped')

def split_corpus(corpus_path):
    """  Shuffle the corpus.

    Split the corpus into 7:2:1 as train, dev, test.
    """
    def dump_corpus_split(lines, filename):
        with open('data/{}'.format(filename), 'w', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(lines), total=len(lines)):
                f.write(line)

    # shuffle corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = f.readlines()
        shuffle(corpus)

    # dump split corpus with raw text
    dump_corpus_split(corpus[:int(len(corpus)*0.7)], 'train.txt')
    dump_corpus_split(corpus[int(len(corpus) * 0.7):int(len(corpus) * 0.9)], 'dev.txt')
    dump_corpus_split(corpus[int(len(corpus) * 0.9):], 'test.txt')

if __name__ == "__main__":
    build_lexicon('data/corpus.txt')
    split_corpus('data/corpus.txt')


