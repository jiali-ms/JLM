import pickle
import numpy as np
import os
import sys
from collections import defaultdict
sys.path.append('..')
from config import root_path

'''
For Japanese or Chinese decoder, reading dictionary is a dict that mapping same pronunciation with different candidates.
It is necessary to build a lattice for decoder to find best path with probabilities.
In this implementation, I the word is in the format of display/reading/POS. where each reading is the key for the dict.
We build it from the lexicon with full vocabulary so that the oov is also included in the dict.  
'''
def build_reading_dict():
    reading_dict = defaultdict(list)
    dict_path = os.path.join(root_path, 'data')
    lexicon = pickle.load(open(os.path.join(dict_path, 'lexicon.pkl'), 'rb'))
    # the lexicon is sorted with frequency already, the index is the same as the w2i.
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
    for reading, l in sorted_reading[:100]:
        print('{}: {}'.format(reading, len(l)))
    pickle.dump(reading_dict, open(os.path.join(dict_path, "reading_dict.pkl"), 'wb'))

if __name__ == "__main__":
    # test the model
    build_reading_dict()
