import time
import tensorflow as tf
from model import RNNLM_Model
import sys
sys.path.append('..')
from config import train_path, experiment_path, ExperimentConfig
from sacred import Experiment
from sacred.observers import FileStorageObserver
ex = Experiment()
ex.observers.append(FileStorageObserver.create("experiments"))

# set parameters here so they can be shared between the experiment observer and the model
parameters = {
    "is_debug": False,
    "batch_size": 128 * 3,
    "embed_size": 256,
    "hidden_size": 512,
    "num_steps": 20,
    "max_epochs": 10,
    "early_stopping": 1,
    "dropout": 0.9,
    "lr": 0.001,
    "share_embedding": True,
    "gpu_id": 0,
    "tf_random_seed": 101,
    "D_softmax": False,
    "V_table": False,
    "embedding_seg": [(200, 0, 6000), (100, 6000, 15000), (50, 15000, None)],
    "tune": False,
    "tune_id": -1,
    "tune_lock_input_embedding": False
}

ex.add_config(parameters)

# specify the GPU to use, 0 is the start index
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters["gpu_id"])

@ex.automain
def train_RNNLM(_run):
    # maintain consistency between sacred config and experiment config
    config = ExperimentConfig(**_run.config)

    experiment_dump_path = os.path.join(experiment_path, str(_run._id), "tf_dump")
    if not os.path.exists(experiment_dump_path):
        os.makedirs(experiment_dump_path)

    if config.is_debug:
        print("Running in debug mode...")

    with tf.Graph().as_default():
        # set random seed before the graph is built
        tf.set_random_seed(config.tf_random_seed)

        model = RNNLM_Model(config)
        model.load_corpus(debug=config.is_debug)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as session:
            best_val_pp = float('inf')
            best_val_epoch = 0

            tf.summary.FileWriter(os.path.join(train_path, "TensorLog"), session.graph)

            session.run(init)
            for epoch in range(config.max_epochs):
                print('Epoch {}'.format(epoch))
                start = time.time()
                train_pp = model.run_epoch(
                    session, model.encoded_train,
                    train_op=model.train_step)
                valid_pp = model.run_epoch(session, model.encoded_valid)
                print('Training perplexity: {}'.format(train_pp))
                print('Total Training time: {}'.format(time.time() - start))
                print('Validation perplexity: {}'.format(valid_pp))
                if valid_pp < best_val_pp:
                    best_val_pp = valid_pp
                    best_val_epoch = epoch
                    saver.save(session, os.path.join(experiment_dump_path, 'rnnlm.weights'))
                if epoch - best_val_epoch > config.early_stopping:
                    break
                print('Total time: {}'.format(time.time() - start))
