from collections import defaultdict
import numpy as np
import pickle
import math
import os
import time
import json
from copy import deepcopy, copy
import sys
import operator
sys.path.append('..')
from config import dict_path, train_path
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

class Decoder():
    def __init__(self):
        self.config = json.loads(open(os.path.join(train_path, str(experiment), "weights", "config.json"), "rt").read())
        self.i2w = pickle.load(open(os.path.join(dict_path, "i2w.pkl"), 'rb'))
        self.w2i = pickle.load(open(os.path.join(dict_path, "w2i.pkl"), 'rb'))
        self.lexicon = pickle.load(open(os.path.join(dict_path, "lexicon.pkl"), 'rb'))
        self.lexicon = {v: k for k, v in self.lexicon.items()}
        self.model = LSTM_Model()
        self.reading_dict = pickle.load(open(os.path.join(dict_path, "reading_dict.pkl"), 'rb'))

    def _parse_reading_id(self, id):
        # id = int(wid) * 100 + int(lex_type) * 10 + int(nolst)
        ignore = id % 10 == 1
        oov = id not in self.i2w.keys()
        return ignore, oov

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
                if sub_token in self.reading_dict.keys():
                    for id in self.reading_dict[sub_token]:
                        ignore, oov = self._parse_reading_id(id)
                        # some words are ignored in best path for performance in existing Japanese IME, align here
                        if ignore:
                            continue
                        word = self.lexicon[id]
                        prob = 0.0
                        if oov:
                            pos = self.lexicon[id].split('/')[-1]
                            unk_word = '<unk_{}>'.format(hex(int(pos))[2:])
                            word = word + '*'  # mark the oov for debug purpose
                            if unk_word not in self.w2i:
                                print(word)
                                continue
                            id = self.w2i[unk_word]
                            # TODO prob = 1 / self.idx2freq[id]  # just share evenly from the distribution.
                        add_node_to_lattice(i, sub_token, id, word, prob)

            # if the input like special token cause no match in dictionary, then add the token directly, mark as <unk>
            if len(backward_lookup[i + 1]) == 0:
                backward_lookup[i + 1].append(Node(i, 1, self.w2i['<unk>'], input[i]))

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
    print("test decoder")
    test_reading_dict = {
        "き": ["き", "期", "気", "祈"],
        "ょ": ["ょ"],
        "きょ": ["きょ", "居", "巨", "許"],
        "きょう": ["きょう", "今日", "京"],
        "きょうの": ["きょうの", "京の"],
        "う": ["う", "右"],
        "は": ["は", "葉", "歯", "派"],
        "い": ["い", "伊", "医", "イ"],
        "いい": ["いい", "言い"],
        "て": ["て", "手"],
        "ん": ["ん"],
        "てん": ["てん", "点", "天"],
        "てんき": ["てんき", "天気", "転機", "テンキ"]
    }

    decoder = Decoder()
    '''
    b = decoder._build_lattice('きょうはいい')
    for key, value in b.items():
        for node in value[:5]:
            print("{}: {}".format(key, node.word))
    '''

    start_time = time.time()
    result = decoder.decode('ていきてきなつういん・かうんせりんぐ', topN=10, beam_width=50, use_oov=True)
    for item in result:
        print(item)
    print("--- %s seconds ---" % (time.time() - start_time))
