from collections import defaultdict
import os
import numpy as np
import sys
import operator

'''
Training corpus that has each line as one sentence. Words are segmented by space. 
'''
def build_training_corpus(bccwj_suw_dir, debug=False):
    corpus_name = 'corpus.txt'
    if debug:
        corpus_name = 'corpus_debug.txt'

    with open('data/' + corpus_name, 'w', encoding='utf-8') as f:
        for folder, subs, files in os.walk(bccwj_suw_dir):
                for filename in files:
                    if '.txt' in filename:
                        path = os.path.join(folder, filename)
                        print('processing file: {}'.format(path))
                        corpus = parse_bccwj_suw(path, debug)
                        for line in corpus:
                            f.write('{}\n'.format(' '.join(line)))

'''     
bccwj corpus disk2 suw folder, contains the segmented corpus.
The key index of the format are
9: 'B' means beginning of sentence
16: POS, e.g. '名詞-普通名詞-一般'
-2: display e.g. '声'
-1: reading in katagana e.g. 'コエ' 

Build a lexicon with key: display/reading/POS value: freq
'''
def parse_bccwj_suw(file_path, debug=False):
    corpus = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            #if i % 1000 == 0:
            #sys.stdout.write('\r progress {0:.2f}'.format(i / len(lines)))
            #sys.stdout.write('\r')
            tokens = line.strip('\n').split('\t')
            try:
                if tokens[9] == 'B' and len(sentence) != 0:
                    corpus.append(sentence)
                    sentence = []

                if tokens[16] == '空白':  # wide space has no reading, exclude from corpus
                    continue

                if debug:
                    word = '{}'.format(tokens[-2], tokens[-1], tokens[16])
                else:
                    word = '{}/{}/{}'.format(tokens[-2], tokens[-1], tokens[16])
                sentence.append(word)
            except:
                print(line)
                pass
    return corpus

build_training_corpus('E:/corpus/BCCWJ/VOL2/SUW', debug=False)