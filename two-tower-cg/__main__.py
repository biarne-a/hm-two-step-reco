from load_data import load_data
from preprocess import preprocess
from train import run_training
from config import Config


def run_all(config: Config):
    # Extract
    train_df, test_df, article_df = load_data()

    # Preprocess
    preprocessed_hm_data = preprocess(train_df, test_df, article_df, batch_size=config.batch_size)

    # Train
    return run_training(preprocessed_hm_data, config)


if __name__ == '__main__':
    config = Config(embedding_dimension=128,
                    batch_size=512,
                    learning_rate=0.05,
                    nb_epochs=4)
    history = run_all(config)
    results = history.history

    import pickle
    pickle.dump(results, open('final_results.p', 'wb'))
