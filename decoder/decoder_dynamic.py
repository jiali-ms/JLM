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
from train.data import Vocab
from decoder import Decoder, Node, Path, CharRNNDecoder

class DynamicDecoder(Decoder):
    """Word based RNN decoder with dynamic decoding.

    Decode with each Kana incrementally.
    """

    def __init__(self, experiment_id=0, comp=0):
        super(DynamicDecoder, self).__init__(experiment_id=experiment_id, comp=comp)
        print('Dynamic RNN decoder loaded')
        self.perf_log_fix_vocab = []  # per log for fix vocabs
        self.perf_log_fix_lattice_path_prob = [] # per step fix lattice cost

    def _build_lattice_vocab(self, backward_lookup, samples=0, top_sampling=False, random_sampling=False):
        self.lattice_vocab = {}

        # vocab frame + top sampling
        def frame_vocab(nodes):
            return sorted([node.word_idx for node in nodes])

        self.lattice_vocab[0] = frame_vocab(backward_lookup[0])

        if samples:
            if random_sampling:
                self.lattice_vocab[0] += [x for x in np.random.randint(len(self.w2i), size=samples)]
            elif top_sampling:
                self.lattice_vocab[0] += [x for x in range(samples)]

        for i in range(1, len(backward_lookup)):
            self.lattice_vocab[i] = sorted(list(set(self.lattice_vocab[i-1]) | set(frame_vocab(backward_lookup[i]))))

            # print('{} vocab selected'.format(len(self.lattice_vocab[i])))

        # print(self.lattice_vocab)


    def _build_current_frame(self, frame, i, beam_width):
        """ Incremental beam search.

            1. find all candidate nodes at current frame
            2. try to connect to previous frames,
            run LSTM model to predict next word prob when prob is not yet calculates
            3. connect and prune to beam size with path prob ranking
        """
        # frame 0 contains one path that has eos_node
        if i == 0:
            frame[0] = [Path(self.backward_lookup[0][0], self.model.hidden_size)]
            return

        # before start, predict the all the necessary prob that current frame requires to build new paths
        self._incremental_decode(frame, i)

        # connect each nodes to its previous best paths
        acc_time = 0
        frame[i] = []
        for node in self.backward_lookup[i]:
            for prev_path in frame[node.start_idx]:
                cur_path = copy(prev_path)  # shallow copy to avoid create dup objects
                cur_path.nodes = copy(prev_path.nodes)
                node_prob_idx = self._frame_vocab(node.start_idx).index(node.word_idx)
                cur_path.append_node(node, node_prob_idx)
                cur_path.prev_path = prev_path
                frame[i].append(cur_path)

                if not self.config['self_norm']:
                    s = time.time()
                    self._fix_neg_log(cur_path)
                    acc_time += (time.time() - s)

        self.perf_log_fix_lattice_path_prob.append(acc_time)

        # prune to beam size
        if beam_width is not None:
            frame[i].sort(key=lambda x: x.neg_log_prob)
            frame[i] = frame[i][:beam_width]

    def _incremental_decode(self, frame, i, max_step=None):
        """Incrementally build path probability distribution.

        A word in current frame may connect to a previous path that has not yet calculated the softmax distribution for
        the word entry.

        E.g. For a coming nodes x-- with length 3, to connect to a previous path, all the frame C needs to update the
        softmax distribution to contain x--.

                           A  B  C  D  E  nodes
                1                x  -  -   x--
                2                   y  -   y-
                3             z  -  -  -   z---
        """
        paths_to_fix = []  # paths to fix in each frame
        batch_missing_vocab = set()  # batch all the missing vocab in all previous connected frame
        frame_diff_vocab = {}  # lookup the diff vocab in a frame

        # fix up all the previous frames.
        for k in range(i):
            # sort the diff to ensure order
            diff_vocab = sorted(set(self.lattice_vocab[i]) - set(self.lattice_vocab[k]))

            if len(diff_vocab):
                # path in i-1 will be batch predicted.
                if k != i-1:
                    paths_to_fix += frame[k]

                frame_diff_vocab[k] = diff_vocab

                # now fix each frame to have same vocab as frame i
                self.lattice_vocab[k] += diff_vocab

                # print('recalculate frame {} with {} words'.format(k, len(vocab_to_extend)))
                batch_missing_vocab |= set(diff_vocab)

        # predict the i-1 frame with fixed vocab
        self._batch_predict(frame[i - 1], self.lattice_vocab[i-1])

        # fix the connection
        if len(paths_to_fix):
            missing_vocab = sorted(list(batch_missing_vocab))

            start_time = time.time()
            logits = self.model.project(np.concatenate([path.state for path in paths_to_fix], axis=0), missing_vocab)
            self.perf_log_fix_vocab.append(time.time() - start_time)

            for i, path in enumerate(paths_to_fix):
                assert len(path.logits)
                diff_vocab = frame_diff_vocab[path.frame_idx]
                diff_vocab_indices = [missing_vocab.index(x) for x in diff_vocab]
                path.logits = np.concatenate((path.logits, logits[i][diff_vocab_indices]))
                if self.config['self_norm']:
                    path.transition_probs = [np.exp(path.logits)]
                else:
                    path.transition_probs = softmax(path.logits)  # recalculate softmax

    def _fix_neg_log(self, path):
        """Fix the path probability by recalculating from head.
        """

        # build a list of path from begin to end
        paths = []
        tail = path
        while tail:
            paths.append(tail)
            tail = tail.prev_path

        paths = paths[::-1]

        s = 1
        max_step = False
        if max_step:
            s = max(s, len(paths) - 5)

        for i in range(s, len(paths)):
            cur_path = paths[i]
            prev_path = paths[i - 1]
            node = cur_path.nodes[-1]
            prob_idx = self.lattice_vocab[node.start_idx].index(node.word_idx)
            new_neg_log_prob = prev_path.neg_log_prob + -math.log(prev_path.transition_probs[0][prob_idx])
            if cur_path.neg_log_prob != new_neg_log_prob:
                cur_path.neg_log_prob = new_neg_log_prob

    def decode(self, input, topN=10, beam_width=10, vocab_select=False, samples=0, top_sampling=False, random_sampling=False):
        self.backward_lookup = self._build_lattice(input, vocab_select=vocab_select, samples=samples, top_sampling=top_sampling, random_sampling=random_sampling)

        frame = {}

        # for each step. t_0 is used for start of sentence
        for i in range(len(input) + 1):
            self._build_current_frame(frame, i, beam_width)

            #for path in frame[i]:
            #    print(path)
                # print(path.state[0][0])

        output = [(x.neg_log_prob, [n.word for n in x.nodes if n.word != "<eos>"]) for x in frame[len(input)]]

        self.perf_sen += 1

        return output[:topN]

if __name__ == "__main__":
    experiment_id = 27
    use_dynamic_decoder = True
    config = json.loads(open(os.path.join(experiment_path, str(experiment_id), 'config.json'), 'rt').read())
    if config['char_rnn']:
        decoder = CharRNNDecoder(experiment_id)
    elif use_dynamic_decoder:
        decoder = DynamicDecoder(experiment_id)
    else:
        decoder = Decoder(experiment_id)

    result = decoder.decode('キョーワイーテンキデス', topN=10, beam_width=10, vocab_select=True, samples=200, top_sampling=False, random_sampling=False)
    for item in result:
        print('{} \t{}'.format(item[0], ' '.join([x.split('/')[0] for x in item[1]])))
    # print(decoder.perf_log)
    print("--- %f seconds lstm per step ---" % (np.mean(decoder.perf_log_lstm)))
    print("--- %f seconds softmax per step ---" % (np.mean(decoder.perf_log_softmax)))
    print("--- %f seconds per step to fix vocab---" % (np.mean(decoder.perf_log_fix_vocab)))
    print("--- %f seconds per step to fix lattice---" % (np.mean(decoder.perf_log_fix_lattice_path_prob)))
    print("--- %f seconds per sentence ---" % (np.sum(decoder.perf_log_lstm + decoder.perf_log_softmax)/decoder.perf_sen))
