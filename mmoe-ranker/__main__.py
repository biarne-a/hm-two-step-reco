import pickle

import pandas as pd

from load_data import load_data
from preprocess import preprocess
from config import Config
from train import run_training


if __name__ == '__main__':
    config = Config(batch_size=512, learning_rate=0.1, nb_epochs=20)
    data = load_data()
    preprocessed_data = preprocess(data, config.batch_size)
    model, history = run_training(preprocessed_data, config)
    pickle.dump(history.history, open('final_results.p', 'wb'))
    model.save('model')
    #
    # model.predict()
