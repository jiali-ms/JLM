from collections import defaultdict
import numpy as np
import pickle
import math
import os
import time
import json
import japanese
from copy import deepcopy, copy
import sys
import operator
sys.path.append('..')
from config import root_path, data_path, train_path, experiment_path, experiment_id
from model import LSTM_Model

class Node():
    def __init__(self, s, l, idx, word, oov_prob=0.0):
        self.start_idx = s  # start index of word
        self.reading_length = l  # length of the reading NOT the word
        self.word_idx = idx  # the index of the node in the output softmax layer
        self.word = word  # word of the node
        self.oov_prob = oov_prob  # for oov word, uni-gram prob under its <unk_pos>

    def __repr__(self):
        return str((self.start_idx, self.word))

class Path():
    """A class that contains all relevant information about each path through the lattice."""
    def __init__(self, node, hidden_size):
        # init with path start '<eos>'
        self.nodes = [node]
        self.cell = np.zeros((1, hidden_size))
        self.state = np.zeros((1, hidden_size))
        self.neg_log_prob = 0.0  # accumulated neg log of the each node to node prob becomes the path neg log prob
        self.transition_probs = []  # note this is not neg log prob

    def append_node(self, node, node_prob_idx):
        self.nodes.append(node)
        if len(self.transition_probs):
            prob = self.transition_probs[0][node_prob_idx]
            if node.oov_prob > 0:
                prob = prob * node.oov_prob  # the prob for unk_pos is shared, need to divide by the word's uni-gram
            self.neg_log_prob += -math.log(prob)

    def __str__(self):
        return ' '.join(['{}'.format(x.word) for x in self.nodes]) + ': {}'.format(self.neg_log_prob)

class Decoder():
    def __init__(self):
        self.config = json.loads(open(os.path.join(experiment_path, str(experiment_id), 'config.json'), 'rt').read())
        self.i2w = pickle.load(open(os.path.join(data_path, 'i2w.pkl'), 'rb'))
        self.w2i = pickle.load(open(os.path.join(data_path, 'w2i.pkl'), 'rb'))

        # full lexicon and reading dictionary covers all the vocab
        # that includes oov words to the model
        self.full_lexicon = pickle.load(open(os.path.join(root_path, 'data', 'lexicon.pkl'), 'rb'))
        self.full_reading_dict = pickle.load(open(os.path.join(root_path, 'data', 'reading_dict.pkl'), 'rb'))

        self.model = LSTM_Model()

    def _check_oov(self, word):
        return word not in self.w2i.keys()

    def _build_lattice(self, input, use_oov=False):
        def add_node_to_lattice(i, sub_token, id, word, prob):
            node = Node(i, len(sub_token), id, word, prob)  # share the node in both lookup table
            backward_lookup[i + len(sub_token)].append(node)

        # backward_lookup keeps the words that ends at a frame
        # e.g. the input A/BC/D, 0 is preserved for <eos>
        # 0        1        2        3         4
        # <eos>    A                BC         D
        backward_lookup = defaultdict(lambda: [])
        eos_node = [Node(-1, 1, self.w2i['<eos>'], '<eos>')]
        backward_lookup[0] = eos_node

        for i, token in enumerate(input):
            for j in range(len(input) - i):
                sub_token = input[i: i + j + 1]
                if sub_token in self.full_reading_dict.keys():
                    for lexicon_id in self.full_reading_dict[sub_token]:
                        word = self.full_lexicon[lexicon_id][0]
                        oov = self._check_oov(word)
                        if oov:
                            # skip oov in this experiment,
                            # note that oov affects conversion quality
                            continue
                        prob = 0.0
                        id = self.w2i[word]
                        add_node_to_lattice(i, sub_token, id, word, prob)

        return backward_lookup

    def _build_current_frame(self, nodes, frame, idx):
        # frame 0 contains one path that has eos_node
        if idx == 0:
            frame[0] = [Path(nodes[0], self.model.hidden_size)]
            return

        # connect each nodes to its previous best paths, also calculate the new path probability
        frame[idx] = []
        for node in nodes:
            for prev_path in frame[node.start_idx]:
                cur_paths = copy(prev_path)  # shallow copy to avoid create dup objects
                cur_paths.nodes = copy(prev_path.nodes)
                node_prob_idx = node.word_idx
                cur_paths.append_node(node, node_prob_idx)
                frame[idx].append(cur_paths)

    def _batch_predict(self, paths):
        pred, state, cell = self.model.predict_with_context([path.nodes[-1].word_idx for path in paths],
                                                            np.concatenate([path.state for path in paths], axis=0),
                                                            np.concatenate([path.cell for path in paths], axis=0))

        for i, path in enumerate(paths):
            path.state = np.expand_dims(state[i], axis=0)
            path.cell = np.expand_dims(cell[i], axis=0)
            path.transition_probs = [pred[i]]

    def decode(self, input, topN=10, beam_width=10, use_oov=False):
        backward_lookup = self._build_lattice(input, use_oov)

        frame = {}
        for i in range(len(input) + 1):
            b_nodes = backward_lookup[i]
            self._build_current_frame(b_nodes, frame, i)

            if beam_width is not None:
                frame[i].sort(key=lambda x: x.neg_log_prob)
                frame[i] = frame[i][:beam_width]

            self._batch_predict(frame[i])

        output = [(x.neg_log_prob, [n.word for n in x.nodes if n.word != "<eos>"]) for x in frame[len(input)]]
        return output[:topN]

if __name__ == "__main__":
    decoder = Decoder()
    start_time = time.time()
    print(input)
    result = decoder.decode('キョーワイーテンキデス', topN=10, beam_width=10, use_oov=True)
    for item in result:
        print(item)
    print("--- %s seconds ---" % (time.time() - start_time))
