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
from config import root_path, data_path, train_path, experiment_path
from model import LSTM_Model, softmax
from train.data import Vocab, CharVocab

class Node():
    def __init__(self, s, l, idx, word, oov_prob=0.0):
        self.start_idx = s  # start index of word
        self.reading_length = l  # length of the reading NOT the word
        self.word_idx = idx  # the index of the node in the output softmax layer
        self.word = word  # word of the node
        self.oov_prob = oov_prob  # for oov word, uni-gram prob under its <unk_pos>
        self.char_rnn_step = 0  # for char RNN, it need to eval prob of full word in several steps

    def __repr__(self):
        return str((self.start_idx, self.word))

class Path():
    """A class that contains all relevant information about each path through the lattice."""
    def __init__(self, node, hidden_size):
        # init with path start '<eos>'
        self.nodes = [node]
        self.cell = np.zeros((1, hidden_size))
        self.state = np.zeros((1, hidden_size))
        self.hidden = None
        self.neg_log_prob = 0.0  # accumulated neg log of the each node to node prob becomes the path neg log prob
        self.transition_probs = []  # note this is not neg log prob
        self.logits = []  # y output from model, cache it to re-calculate softmax
        self.prev_path = None
        self.frame_idx = self.nodes[-1].start_idx + self.nodes[-1].reading_length

    def append_node(self, node, node_prob_idx):
        self.nodes.append(node)
        if len(self.transition_probs):
            prob = self.transition_probs[0][node_prob_idx]
            if node.oov_prob > 0:
                prob = prob * node.oov_prob  # the prob for unk_pos is shared, need to divide by the word's uni-gram
            self.neg_log_prob += -math.log(prob)

    def __str__(self):
        return ' '.join(['{}'.format(x.word.split('/')[0]) for x in self.nodes]) + ': {}'.format(self.neg_log_prob)

class Decoder():
    def __init__(self, experiment_id=0, comp=0):
        self.config = json.loads(open(os.path.join(experiment_path, str(experiment_id), 'config.json'), 'rt').read())
        self._load_vocab()

        # full lexicon and reading dictionary covers all the vocab
        # that includes oov words to the model
        self.full_lexicon = pickle.load(open(os.path.join(root_path, 'data', 'lexicon.pkl'), 'rb'))
        self.full_reading_dict = pickle.load(open(os.path.join(root_path, 'data', 'reading_dict.pkl'), 'rb'))

        self.model = LSTM_Model(experiment_id, comp)

        self.lattice_vocab = None
        self.perf_sen = 0
        self.perf_log_lstm = []
        self.perf_log_softmax = []

    def _load_vocab(self):
        self.vocab = Vocab(self.config['vocab_size'])
        self.i2w = self.vocab.i2w
        self.w2i = self.vocab.w2i

    def _check_oov(self, word):
        return word not in self.w2i

    def _build_lattice(self, input, vocab_select=False, samples=0, top_sampling=False, random_sampling=False):
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
                    word_set = set()
                    for lexicon_id in sorted(self.full_reading_dict[sub_token]):
                        word = self.full_lexicon[lexicon_id][0]
                        oov = self._check_oov(word)
                        if oov:
                            # skip oov in this experiment,
                            # note that oov affects conversion quality
                            continue

                        prob = 0.0
                        if self.config['char_rnn']:
                            oov = self._char_check_oov(word)
                            if oov:
                                # skip oov in this experiment,
                                # note that oov affects conversion quality
                                continue

                            # char rnn case, we still build lattice use word, keep the first char as index of the node
                            word = word.split('/')[0]

                            if word in word_set:
                                continue

                            if len(word_set) > 200:
                                continue

                            word_set.add(word)
                            id = self.w2i[word[0]]
                        else:
                            id = self.w2i[word]
                        add_node_to_lattice(i, sub_token, id, word, prob)

                # put symbol directly if no match at all in reading dictionary
                if len(backward_lookup[i + 1]) == 0:
                    backward_lookup[i + 1].append(Node(i, 1, self.w2i['<unk>'], input[i]))

        if vocab_select:
            self._build_lattice_vocab(backward_lookup, samples, top_sampling, random_sampling)

        return backward_lookup

    def _build_lattice_vocab(self, backward_lookup, samples=0, top_sampling=False, random_sampling=False):
        # vocab selection is an advanced pruning algorithm to limit the vocab space in a specified
        # conversion task. It will avoid calculating softmax for the full vocab.
        def lattice_vocab(backward_lookup):
            return sorted(list(set(sum([[node.word_idx for node in nodes] for nodes in backward_lookup.values()], []))))

        self.lattice_vocab = lattice_vocab(backward_lookup)

        if samples:
            if random_sampling:
                samples = np.random.randint(len(self.w2i), size=samples)
                self.lattice_vocab += [x for x in samples]
            elif top_sampling:
                self.lattice_vocab += [x for x in range(samples)]
            self.lattice_vocab = sorted(list(set(self.lattice_vocab)))

        #if self.lattice_vocab:
        #    print('{} vocab selected'.format(len(self.lattice_vocab)))

    def _frame_vocab(self, frame):
        if type(self.lattice_vocab) is list:
            return self.lattice_vocab
        elif type(self.lattice_vocab) is dict:
            return self.lattice_vocab[frame]
        else:
            return None

    def _build_current_frame(self, frame, idx):
        # frame 0 contains one path that has eos_node
        if idx == 0:
            frame[0] = [Path(self.backward_lookup[0][0], self.model.hidden_size)]
            return

        # connect each nodes to its previous best paths, also calculate the new path probability
        frame[idx] = []
        for node in self.backward_lookup[idx]:
            for prev_path in frame[node.start_idx]:
                # shallow copy to avoid create dup objects
                # note that numpy arrays like state and cell are copied also
                cur_path = copy(prev_path)
                cur_path.nodes = copy(prev_path.nodes)
                node_prob_idx = node.word_idx
                if self.lattice_vocab:
                    node_prob_idx = self._frame_vocab(idx).index(node_prob_idx)
                cur_path.append_node(node, node_prob_idx)
                frame[idx].append(cur_path)

    def _predict(self, paths, vocab=None):
        log_lstm = []
        log_softmax = []
        for i, path in enumerate(paths):
            (p, logits, log_lstm_time, log_softmax_time), s, c = self.model.predict_with_context([path.nodes[-1].word_idx],
                                                      path.state,
                                                      path.cell,
                                                      vocab)
            log_lstm.append(log_lstm_time)
            log_softmax.append(log_softmax_time)

            path.state = s
            path.cell = c
            path.transition_probs = p

        self.perf_log_lstm.append(sum(log_lstm))
        self.perf_log_softmax.append(sum(log_softmax))

    def _batch_predict(self, paths, vocab=None):
        # print('batch predict paths {}'.format(len(paths)))
        start_time = time.time()

        (pred, logits, log_lstm_time, log_softmax_time), state, cell = self.model.predict_with_context([path.nodes[-1].word_idx for path in paths],
                                                            np.concatenate([path.state for path in paths], axis=0),
                                                            np.concatenate([path.cell for path in paths], axis=0),
                                                            vocab)

        self.perf_log_lstm.append(log_lstm_time)
        self.perf_log_softmax.append(log_softmax_time)

        for i, path in enumerate(paths):
            path.state = np.expand_dims(state[i], axis=0)
            path.cell = np.expand_dims(cell[i], axis=0)
            path.transition_probs = [pred[i]]
            path.logits = logits[i]

    def decode(self, input, topN=10, beam_width=10, vocab_select=False, samples=0, top_sampling=False, random_sampling=False):
        self.backward_lookup = self._build_lattice(input, vocab_select=vocab_select, samples=samples, top_sampling=top_sampling, random_sampling=random_sampling)

        frame = {}
        for i in range(len(input) + 1):
            self._build_current_frame(frame, i)

            if beam_width is not None:
                frame[i].sort(key=lambda x: x.neg_log_prob)
                frame[i] = frame[i][:beam_width]

            batch_predict = True
            if batch_predict:
                self._batch_predict(frame[i], self._frame_vocab(i))
            else:
                self._predict(frame[i])

        output = [(x.neg_log_prob, [n.word for n in x.nodes if n.word != "<eos>"]) for x in frame[len(input)]]

        self.perf_sen += 1

        return output[:topN]


class CharRNNDecoder(Decoder):
    """Char RNN based decoder.

    Since there is no easy to to build a lattice per char, use words in the lexicon to build lattice.
    To decode the word based lattice using char model, we take the probability directly if the node has only one char.
    Otherwise, we will run a full evaluation of p([c1, c2, ..., cn]|context), where word is [c1, c2, ... ,cn].

                                                backward lookup |
                                                      C11 C12 C13  <-- need eval over full word
           prev_path (distribution for Cx1)                   C21  <-- use softmax prob directly
                                                          C31 C32  <-- need eval over full word

    Batching the eval set for each step is still possible, leave this as TODO if necessary
    """

    def __init__(self, experiment_id=0, comp=0):
        super(CharRNNDecoder, self).__init__(experiment_id=experiment_id, comp=comp)
        print('Char RNN decoder loaded')

    def _check_oov(self, word):
        return word not in self.vocab.words

    def _char_check_oov(self, word):
        return sum([c not in self.w2i for c in word.split('/')[0]])

    def _word_length(self, word):
        if word == '<eos>' or word == '<unk>':
            return 1
        else:
            return len(word)


    def _build_current_frame(self, nodes, frame, idx):
        # frame 0 contains one path that has eos_node
        if idx == 0:
            frame[0] = [Path(nodes[0], self.model.hidden_size)]
            return

        # connect each nodes to its previous best paths, also calculate the new path probability
        frame[idx] = []
        unique_path = set()
        for node in nodes:
            for prev_path in frame[node.start_idx]:
                cur_path = copy(prev_path)  # shallow copy to avoid create dup objects
                cur_path.nodes = copy(prev_path.nodes)
                cur_node = copy(node)
                node_prob_idx = node.word_idx
                cur_path.append_node(cur_node, node_prob_idx)

                string_path = ''.join([x.word for x in cur_path.nodes])
                if string_path not in unique_path:
                    # print(string_path)
                    unique_path.add(string_path)
                    frame[idx].append(cur_path)

        #print('frame size {}'.format(len(frame[idx])))

    def _eval_frame(self, paths):
        temp_batch = []
        for path in paths:
            n = path.nodes[-1]
            if n.char_rnn_step + 1 < self._word_length(n.word):
                temp_batch.append(path)

        if len(temp_batch) > 0:
            #print('{} eval'.format(len(temp_batch)))

            self._batch_predict(temp_batch)
            for path in temp_batch:
                n = path.nodes[-1]
                n.char_rnn_step += 1
                #if n.char_rnn_step + 1 < self._word_length(n.word):
                n.word_idx = self.w2i[n.word[n.char_rnn_step]]
                path.neg_log_prob += -np.log(path.transition_probs[0][n.word_idx])

            # recursively eval till end, this is not very efficient if a long word is the only one item
            self._eval_frame(temp_batch)

    def decode(self, input, topN=10, beam_width=10, vocab_select=False, samples=0, top_sampling=False, random_sampling=False):
        backward_lookup = self._build_lattice(input, vocab_select=vocab_select, samples=samples, top_sampling=top_sampling, random_sampling=random_sampling)

        frame = {}
        for i in range(len(input) + 1):
            b_nodes = backward_lookup[i]
            self._build_current_frame(b_nodes, frame, i)
            self._eval_frame(frame[i])

            if beam_width is not None:
                frame[i].sort(key=lambda x: x.neg_log_prob)
                frame[i] = frame[i][:beam_width]

            self._batch_predict(frame[i])

        output = [(x.neg_log_prob, [n.word for n in x.nodes if n.word != "<eos>"]) for x in frame[len(input)]]

        self.perf_sen += 1

        return output[:topN]

if __name__ == "__main__":
    experiment_id = 27
    config = json.loads(open(os.path.join(experiment_path, str(experiment_id), 'config.json'), 'rt').read())
    if config['char_rnn']:
        decoder = CharRNNDecoder(experiment_id)
    else:
        decoder = Decoder(experiment_id)

    result = decoder.decode('キョーワイーテンキデス', topN=10, beam_width=10, vocab_select=True, samples=0, top_sampling=False, random_sampling=False)
    for item in result:
        print('{} \t{}'.format(item[0], ' '.join([x.split('/')[0] for x in item[1]])))

    print("--- %s seconds lstm per step ---" % (np.mean(decoder.perf_log_lstm)))
    print("--- %s seconds softmax per step ---" % (np.mean(decoder.perf_log_softmax)))
    print("--- %s seconds per sentence ---" % (np.sum(decoder.perf_log_lstm + decoder.perf_log_softmax) /decoder.perf_sen))
