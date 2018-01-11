import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import sequence_loss
from utils import *
sys.path.append('..')
from config import get_configs, experiment_path, data_path

class Vocab(object):
    def __init__(self):
        print("init vocab...")
        self.w2i = pickle.load(open(os.path.join(data_path, 'w2i.pkl'), 'rb'))
        self.i2w = pickle.load(open(os.path.join(data_path, 'i2w.pkl'), 'rb'))
        self.lexicon = pickle.load(open(os.path.join(data_path, 'lexicon.pkl'), 'rb'))

    def encode(self, word):
        return self.w2i[word]

    def decode(self, index):
        return self.i2w[index]

    def __len__(self):
        return len(self.lexicon)

class Corpus(object):
    def __init__(self, debug=False):
        print("load corpus...")
        self.encoded_corpus = pickle.load(open(os.path.join(data_path, 'encoded_corpus.pkl'), 'rb'))
        print("corpus loaded {} words".format(len(self.encoded_corpus)))

        self.size = len(self.encoded_corpus)
        if debug:
            self.size = 1024 * 1024

    def encoded_train(self):
        return self.encoded_corpus[:round(self.size * 0.7)]

    def encoded_valid(self):
        return self.encoded_corpus[round(self.size * 0.7): round(self.size * 0.9)]

    def encoded_test(self):
        return self.encoded_corpus[round(self.size * 0.9): self.size]


# the RNNLM code borrowed sample code from http://cs224d.stanford.edu/assignment2/index.html
class RNNLM_Model():
    def __init__(self, config):
        # init
        print('init training model...')
        self.config = config
        self.load_dict()
        self.load_corpus()
        if self.config.D_softmax:
            self.build_D_softmax_mask()
        if self.config.V_table:
            self.build_embedding_from_V_tables()

        # define model graph
        self.add_placeholders()
        inputs = self.add_embedding()
        rnn_outputs = self.add_lstm_model(inputs)
        projection_outputs = self.add_projection(rnn_outputs)

        # Cast o to float64 as there are numerical issues (i.e. sum(output of softmax) = 1.00000298179 and not 1)
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in projection_outputs]
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as needed to evenly divide
        output = tf.reshape(tf.concat(projection_outputs, 1), [-1, len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)

    def build_embedding_from_V_tables(self):
        with tf.variable_scope('embedding'):
            # variable table is a variation to differentiated softmax. Instead of concat embeddings from different zone,
            # it uses matrix factorization to decompose each zone |V1| x e into |V1| x e1 and e1 x e.
            # Implementation wise, we define both matrix at initialization and train them all together.
            self.blocks = []
            self.v_tables = []
            embeddings = []
            self.config.embed_size = self.config.embedding_seg[0][0]
            for i, (size, s, e) in enumerate(self.config.embedding_seg):
                if e is None:
                    e = len(self.vocab)
                block = tf.get_variable('LM{}'.format(i), [e-s, size])
                self.blocks.append(block)
                if i != 0:
                    table = tf.get_variable('VT{}'.format(i), [size, self.config.embed_size])
                    self.v_tables.append(table)
                    embeddings.append(tf.matmul(block, table))
                else:
                    embeddings.append(block)

            self.v_table_embedding = tf.concat(embeddings, axis=0)

    def build_D_softmax_mask(self):
        # differentiated softmax will split the vocab in different segmentation
        # each segmentation has a unique embedding size, e.g.
        # A 0 0
        # 0 B 0
        # 0 0 C
        # block A (|Va| x 200), B (|Vb| x 100), C (|Vc| x 50), the embedding size is 200 + 100 + 50
        # the rest of the parts are filled with zeros and should not be used.
        # block A, B, C are essentially different embedding matrices, putting them together is for simple lookup
        self.config.embed_size = sum([x[0] for x in self.config.embedding_seg])
        D_softmax_mask = np.zeros((len(self.vocab), self.config.embed_size))
        col_s = 0
        for size, s, e in self.config.embedding_seg:
            if e is None:
                e = len(self.vocab)
            # fill each block with ones as mask
            D_softmax_mask[s:e, col_s:col_s + size] = np.ones((e - s, size))
            col_s += size
        self.D_softmax_mask = tf.constant(D_softmax_mask, dtype=tf.float32)

    def load_corpus(self, debug=False):
        self.corpus = Corpus(debug)
        self.encoded_train = np.array(self.corpus.encoded_train())
        self.encoded_valid = np.array(self.corpus.encoded_valid())
        self.encoded_test = np.array(self.corpus.encoded_test())

    def load_dict(self):
        self.vocab = Vocab()

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
        self.dropout_placeholder = tf.placeholder(tf.float32, None)

    def add_embedding(self):
        with tf.variable_scope('embedding'):
            if self.config.tune:
                if self.config.tune_lock_input_embedding:
                    embedding = tf.get_variable('LM', initializer=tf.constant(self.LM.astype(np.float32)),
                                                trainable=False)
                else:
                    embedding = tf.get_variable('LM', initializer=tf.constant(self.weights['LM'].astype(np.float32)))
            else:
                embedding = tf.get_variable('LM', [len(self.vocab), self.config.embed_size])

            if self.config.V_table:
                embedding = self.v_table_embedding
            elif self.config.D_softmax:
                # use mask to keep only the blocks in this patched embedding matrix
                embedding = tf.multiply(embedding, self.D_softmax_mask)

            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            inputs = [tf.squeeze(x, [1]) for x in tf.split(inputs, self.config.num_steps, 1)]
            return inputs

    def add_lstm_model(self, inputs):
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]

        with tf.variable_scope('LSTM') as scope:
            self.initial_state = tf.zeros(
                [self.config.batch_size, self.config.hidden_size])
            self.initial_cell = tf.zeros(
                [self.config.batch_size, self.config.hidden_size])
            state = self.initial_state
            cell = self.initial_cell
            rnn_outputs = []
            for tstep, current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                # input, forget, output, gate
                if self.config.tune:  # init from pre-trained results
                    Hi = tf.get_variable('HMi', initializer=tf.constant(self.weights['HMi'].astype(np.float32)))
                    Hf = tf.get_variable('HMf', initializer=tf.constant(self.weights['HMf'].astype(np.float32)))
                    Ho = tf.get_variable('HMo', initializer=tf.constant(self.weights['HMo'].astype(np.float32)))
                    Hg = tf.get_variable('HMg', initializer=tf.constant(self.weights['HMg'].astype(np.float32)))

                    Ii = tf.get_variable('IMi', initializer=tf.constant(self.weights['IMi'].astype(np.float32)))
                    If = tf.get_variable('IMf', initializer=tf.constant(self.weights['IMf'].astype(np.float32)))
                    Io = tf.get_variable('IMo', initializer=tf.constant(self.weights['IMo'].astype(np.float32)))
                    Ig = tf.get_variable('IMg', initializer=tf.constant(self.weights['IMg'].astype(np.float32)))

                    bi = tf.get_variable('bi', initializer=tf.constant(self.weights['bi'].astype(np.float32)))
                    bf = tf.get_variable('bf', initializer=tf.constant(self.weights['bf'].astype(np.float32)))
                    bo = tf.get_variable('bo', initializer=tf.constant(self.weights['bo'].astype(np.float32)))
                    bg = tf.get_variable('bg', initializer=tf.constant(self.weights['bg'].astype(np.float32)))
                else:
                    Hi = tf.get_variable('HMi', [self.config.hidden_size, self.config.hidden_size])
                    Hf = tf.get_variable('HMf', [self.config.hidden_size, self.config.hidden_size])
                    Ho = tf.get_variable('HMo', [self.config.hidden_size, self.config.hidden_size])
                    Hg = tf.get_variable('HMg', [self.config.hidden_size, self.config.hidden_size])

                    Ii = tf.get_variable('IMi', [self.config.embed_size, self.config.hidden_size])
                    If = tf.get_variable('IMf', [self.config.embed_size, self.config.hidden_size])
                    Io = tf.get_variable('IMo', [self.config.embed_size, self.config.hidden_size])
                    Ig = tf.get_variable('IMg', [self.config.embed_size, self.config.hidden_size])

                    bi = tf.get_variable('bi', [self.config.hidden_size])
                    bf = tf.get_variable('bf', [self.config.hidden_size])
                    bo = tf.get_variable('bo', [self.config.hidden_size])
                    bg = tf.get_variable('bg', [self.config.hidden_size])

                i = tf.nn.sigmoid(tf.matmul(state, Hi) + tf.matmul(current_input, Ii) + bi)
                f = tf.nn.sigmoid(tf.matmul(state, Hf) + tf.matmul(current_input, If) + bf)
                o = tf.nn.sigmoid(tf.matmul(state, Ho) + tf.matmul(current_input, Io) + bo)
                g = tf.nn.tanh(tf.matmul(state, Hg) + tf.matmul(current_input, Ig) + bg)

                cell = tf.multiply(cell, f) + tf.multiply(g, i)
                state = tf.multiply(tf.nn.tanh(cell), o)

                rnn_outputs.append(state)

            self.final_state = rnn_outputs[-1]

        with tf.variable_scope('RNNDropout'):
            rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in rnn_outputs]

        return rnn_outputs

    def add_projection(self, rnn_outputs):
        # share the input and output embedding to save the model size
        # note that it is not part of standard LSTM model,
        # it requires a special projection between hidden and output embedding layer

        if self.config.tune:
            print('embedding is pre-trained or compressed. Now locked it to tune rest of the model')
            embedding = tf.get_variable('CLM', initializer=tf.constant(self.LM.astype(np.float32)), trainable=False)
        else:
            if self.config.share_embedding:
                with tf.variable_scope('embedding', reuse=True):
                    embedding = tf.get_variable('LM')

            if self.config.V_table:
                embedding = self.v_table_embedding
            elif self.config.D_softmax:
                embedding = tf.multiply(embedding, self.D_softmax_mask)

        with tf.variable_scope('Projection'):
            if self.config.share_embedding:
                if self.config.tune:
                    P = tf.get_variable('PM', initializer = tf.constant(self.weights['PM'].astype(np.float32)))
                    U = tf.matmul(P, tf.transpose(embedding))
                else:
                    P = tf.get_variable('PM', [self.config.hidden_size, self.config.embed_size])
                    U = tf.matmul(P, tf.transpose(embedding))
            else:
                U = tf.get_variable('UM', [self.config.hidden_size, len(self.vocab)])

            if self.config.tune:
                proj_b = tf.get_variable('b2', initializer = tf.constant(self.weights['b2'].astype(np.float32)))
            else:
                proj_b = tf.get_variable('b2', [len(self.vocab)])
            outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]

        return outputs

    def add_loss_op(self, output):
        with tf.name_scope('losses'):
            all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
            # sequence loss is the mean of batch and sentence loss
            cross_entropy = sequence_loss([output], [tf.reshape(self.labels_placeholder, [-1])], all_ones, len(self.vocab))
            return cross_entropy

    def add_training_op(self, loss):
        with tf.variable_scope("training", reuse=None):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            train_op = optimizer.minimize(loss)
        return train_op

    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in corpus_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        cell = self.initial_cell.eval()
        for step, (x, y) in enumerate(
                corpus_iterator(data, config.batch_size, config.num_steps)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.initial_state: state,
                    self.initial_cell: cell,
                    self.dropout_placeholder: dp}

            loss, state, _ = session.run(
                [self.calculate_loss, self.final_state, train_op], feed_dict=feed)

            # check each loss to make sure repro
            if step < 10:
                print(loss)

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))

if __name__ == "__main__":
    print("")