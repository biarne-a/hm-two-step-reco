from load_data import load_data
from preprocess import preprocess
from config import Config
from train import run_training


if __name__ == '__main__':
    config = Config(batch_size=512, learning_rate=0.1, nb_epochs=100)
    train_df, test_df, all_variables = load_data()
    data = preprocess(train_df, test_df, config.batch_size, all_variables)
    run_training(data, config)

    # import pickle
    #
    # train_df = pickle.load(open('train_df.p', 'rb'))
    # test_df = pickle.load(open('test_df.p', 'rb'))
    #
    # train_small_df = train_df.sample(frac=0.5)
    # test_small_df = test_df.sample(frac=0.5)
    #
    # pickle.dump(train_small_df, open('train_small_df.p', 'wb'))
    # pickle.dump(test_small_df, open('test_small_df.p', 'wb'))
