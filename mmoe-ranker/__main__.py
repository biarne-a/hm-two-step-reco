import pickle
from load_data import load_data
from preprocess import preprocess
from config import Config
from train import run_training


if __name__ == '__main__':
    config = Config(batch_size=512, learning_rate=0.1, nb_epochs=10)
    data = load_data()
    data = preprocess(data, config.batch_size)
    history = run_training(data, config)
    pickle.dump(history.history, open('final_results.p', 'wb'))
