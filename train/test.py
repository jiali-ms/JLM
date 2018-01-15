import os
from model import RNNLM_Model
import tensorflow as tf
from utils import *
import sys
sys.path.append('..')
from config import experiment_path, get_configs


def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
    state = model.initial_state.eval()
    cell = model.initial_cell.eval()
    # Imagine tokens as a batch size of one, length of len(tokens[0])
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    for i in range(stop_length):
        feed = {model.input_placeholder: [tokens[-1:]],
                model.initial_state: state,
                model.initial_cell: cell,
                model.dropout_placeholder: 1}
        state, y_pred = session.run(
            [model.final_state, model.predictions[-1]], feed_dict=feed)

        next_word_idx = sample(y_pred[0], temperature=temp)
        tokens.append(next_word_idx)
        if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
            break
    output = [model.vocab.decode(word_idx) for word_idx in tokens]
    return output

def generate_sentence(session, model, config, *args, **kwargs):
    return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def auto_generate_sentence(experiment=1):
    gen_config = get_configs(experiment)
    gen_config.batch_size = gen_config.num_steps = 1

    gen_model = RNNLM_Model(gen_config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        saver.restore(session, os.path.join(experiment_path, str(experiment), "tf_dump", 'rnnlm.weights'))
        starting_text = '<eos>'
        while starting_text:
            print(' '.join(generate_sentence(
                session, gen_model, gen_config, starting_text=starting_text, temp=1.0)))
            starting_text = input('> ')

if __name__ == "__main__":
    auto_generate_sentence(experiment=2)