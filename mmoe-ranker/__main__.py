import os
import pickle
import random

import numpy as np
import tensorflow as tf

from load_data import load_data
from preprocess import preprocess
from config import Config
from train import run_training


def set_seed(seed):
    tf.random.set_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':
    set_seed(42)
    config = Config(batch_size=512, learning_rate=0.01, nb_epochs=30)
    data = load_data()
    preprocessed_data = preprocess(data, config.batch_size)
    model, history = run_training(preprocessed_data, config)
    pickle.dump(history.history, open('final_results.p', 'wb'))
    model.save('model')
