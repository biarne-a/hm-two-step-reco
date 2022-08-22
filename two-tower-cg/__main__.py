from load_data import load_data
from preprocess import preprocess
from train import run_training
from config import Config


def run_all(config: Config):
    # Extract
    # hm_data = load_data()

    import pickle
    # pickle.dump(hm_data, open('hm_data.p', 'wb'))
    hm_data = pickle.load(open('hm_data.p', 'rb'))

    # Preprocess
    preprocessed_hm_data = preprocess(hm_data, batch_size=config.batch_size)

    # Train
    run_training(preprocessed_hm_data, config)


if __name__ == '__main__':
    # Define config
    config = Config(embedding_dimension=128,
                    batch_size=512,
                    learning_rate=0.01,
                    nb_epochs=4)

    run_all(config)
