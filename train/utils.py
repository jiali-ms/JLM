import numpy as np

def calculate_perplexity(log_probs):
    # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    perp = 0
    for p in log_probs:
        perp += -p
    return np.exp(perp / len(log_probs))

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def corpus_iterator(raw_data, batch_size, num_steps):
    # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)

import os
def temp():
    for folder, subs, files in os.walk('E:/JLM/train/experiments'):
            for filename in files:
                if 'cout.txt' in filename:
                    print(folder)
                    path = os.path.join(folder, filename)
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if 'Validation perplexity' in line:
                                print(line.strip('\n').split(':')[1].strip())
temp()