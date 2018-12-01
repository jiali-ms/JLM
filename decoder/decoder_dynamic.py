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

    def _build_current_frame(self, nodes, frame, idx):
        # frame 0 contains one path that has eos_node
        if idx == 0:
            frame[0] = [Path(nodes[0], self.model.hidden_size)]
            return

        # connect each nodes to its previous best paths, also calculate the new path probability
        frame[idx] = []
        for node in nodes:
            for prev_path in frame[node.start_idx]:
                cur_path = copy(prev_path)  # shallow copy to avoid create dup objects
                cur_path.nodes = copy(prev_path.nodes)
                node_prob_idx = self._frame_vocab(idx).index(node.word_idx)
                cur_path.append_node(node, node_prob_idx)
                cur_path.prev_path = prev_path
                frame[idx].append(cur_path)

                self._fix_neg_log(cur_path)

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

        for i in range(1, len(paths)):
            cur_path = paths[i]
            prev_path = paths[i-1]
            node = cur_path.nodes[-1]
            prob_idx = self.lattice_vocab[node.start_idx].index(node.word_idx)
            new_neg_log_prob = prev_path.neg_log_prob + -math.log(prev_path.transition_probs[0][prob_idx])
            if cur_path.neg_log_prob != new_neg_log_prob:
                cur_path.neg_log_prob = new_neg_log_prob

    def _dynamic_batch_predict(self, frame, i):
        """Incrementally build path probability.

        To build a new frame, all the paths need to be connected.

                           A  B  C  D  E  nodes
                1                x  -  -   x--
                2                   y  -   y-
                3             z  -  -  -   z---

        We need to give a vocab for each frame to calculate.
        For the previous frame, the vocab is the with accumulated lattice words.
        For the other path to connect, only the missing vocab needs to be append and recalculate softmax.
        """
        # print('------build frame {}'.format(i))

        # calculate the previous first
        # beam size * vocab
        # print('batch predict frame  size {}'.format(len(frame[i-1])))
        self._batch_predict(frame[i-1], self.lattice_vocab[i])
        self.lattice_vocab[i-1] = self.lattice_vocab[i]

        # fix the missing probabilities
        paths = []
        path2frame = []
        missing_vocab = set()  # batch all the missing vocab in all previous connected frame
        diff_vocab_frame = {}

        # fix up all the previous frames
        for k in range(i-1):
            diff_vocab = set(self.lattice_vocab[i]) - set(self.lattice_vocab[k])
            if len(diff_vocab):
                paths += frame[k]
                path2frame += [k] * len(frame[k])
                vocab_to_extend = sorted(list(diff_vocab))
                diff_vocab_frame[k] = vocab_to_extend
                self.lattice_vocab[k] += vocab_to_extend  # extend only

                # print('recalculate frame {} with {} words'.format(k, len(vocab_to_extend)))
                missing_vocab |= diff_vocab

        if len(paths):
            missing_vocab = sorted(list(missing_vocab))

            start_time = time.time()
            logits = self.model.project(np.concatenate([path.state for path in paths], axis=0), missing_vocab)
            self.perf_log_fix_vocab.append(time.time() - start_time)

            for i, path in enumerate(paths):
                assert len(path.logits)
                diff_vocab = diff_vocab_frame[path2frame[i]]
                diff_vocab_indices = [missing_vocab.index(x) for x in diff_vocab]
                path.logits = np.concatenate((path.logits, logits[i][diff_vocab_indices]))
                path.transition_probs = softmax(path.logits)  # recalculate softmax


    def _dynamic_batch_predict2(self, frame, i):
        """ This version recalculate the previous paths.
        """
        
        print('---------non-incremental predict used')
        
        # print('------build frame {}'.format(i))

        # calculate the previous first
        # beam size * vocab
        # print('batch predict frame  size {}'.format(len(frame[i-1])))
        self._batch_predict(frame[i-1], self.lattice_vocab[i])
        self.lattice_vocab[i-1] = self.lattice_vocab[i]

        # fix the missing probabilities
        paths = []
        path2frame = []

        # fix up all the previous frames
        for k in range(i-1):
            diff_vocab = set(self.lattice_vocab[i]) - set(self.lattice_vocab[k])
            #if len(diff_vocab):
            paths += frame[k]
            path2frame += [k] * len(frame[k])
            self.lattice_vocab[k] = self.lattice_vocab[i]

        if len(paths):
            (pred, logits, log_lstm_time, log_softmax_time), state, cell = self.model.predict_with_context([path.nodes[-1].word_idx for path in paths],
                                                                          np.concatenate([path.prev_path.state if path.prev_path else np.zeros(path.state.shape) for path in paths],
                                                                                         axis=0),
                                                                          np.concatenate([path.prev_path.cell if path.prev_path else np.zeros(path.cell.shape) for path in paths],
                                                                                         axis=0),
                                                                          self.lattice_vocab[i])

            self.perf_log_fix_vocab.append(log_softmax_time + log_lstm_time)

            for i, path in enumerate(paths):
                assert len(path.logits)
                path.logits = logits[i]
                path.transition_probs = [pred[i]]

    def decode(self, input, topN=10, beam_width=10, vocab_select=False, samples=0, top_sampling=False, random_sampling=False):
        backward_lookup = self._build_lattice(input, vocab_select=vocab_select, samples=samples, top_sampling=top_sampling, random_sampling=random_sampling)

        frame = {}
        self._build_current_frame(backward_lookup[0], frame, 0)

        for i in range(1, len(input) + 1):
            # calculate probability distribution of previous frames
            # so that nodes in current frame can connect to these paths and have path probability
            self._dynamic_batch_predict(frame, i)

            self._build_current_frame(backward_lookup[i], frame, i)

            if beam_width is not None:
                frame[i].sort(key=lambda x: x.neg_log_prob)
                frame[i] = frame[i][:beam_width]

        output = [(x.neg_log_prob, [n.word for n in x.nodes if n.word != "<eos>"]) for x in frame[len(input)]]

        self.perf_sen += 1

        return output[:topN]

if __name__ == "__main__":
    experiment_id = 8
    use_dynamic_decoder = True
    config = json.loads(open(os.path.join(experiment_path, str(experiment_id), 'config.json'), 'rt').read())
    if config['char_rnn']:
        decoder = CharRNNDecoder(experiment_id)
    elif use_dynamic_decoder:
        decoder = DynamicDecoder(experiment_id)
    else:
        decoder = Decoder(experiment_id)

    result = decoder.decode('キョーワイーテンキデス', topN=10, beam_width=10, vocab_select=True, samples=200, top_sampling=True, random_sampling=False)
    for item in result:
        print('{} \t{}'.format(item[0], ' '.join([x.split('/')[0] for x in item[1]])))
    # print(decoder.perf_log)
    print("--- %s seconds lstm per step ---" % (np.mean(decoder.perf_log_lstm)))
    print("--- %s seconds softmax per step ---" % (np.mean(decoder.perf_log_softmax)))
    print("--- %s seconds per step to fix vocab---" % (np.mean(decoder.perf_log_fix_vocab)))
    print("--- %s seconds per sentence ---" % (np.sum(decoder.perf_log_lstm + decoder.perf_log_softmax)/decoder.perf_sen))
