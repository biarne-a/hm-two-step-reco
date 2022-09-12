import pickle

import pandas as pd

from load_data import load_data
from preprocess import preprocess
from config import Config
from train import run_training


def save_extract(preprocessed_data):
    batch_x, batch_y = next(iter(preprocessed_data.train_ds.take(1000*512).batch(1000*512)))
    batch = {}
    for key, tensor_value in batch_x.items():
        batch[key] = tensor_value.numpy().ravel()
    train_df = pd.DataFrame.from_dict(batch)
    train_df.to_csv('train_df.csv')


if __name__ == '__main__':
    config = Config(batch_size=512, learning_rate=0.1, nb_epochs=10)
    data = load_data()
    preprocessed_data = preprocess(data, config.batch_size)
    model, history = run_training(preprocessed_data, config)
    pickle.dump(history.history, open('final_results.p', 'wb'))
    model.save('model')
    #
    # model.predict()
