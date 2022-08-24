from load_data import load_data
from preprocess import preprocess
from train import run_training
from config import Config


def run_all(config: Config):
    # Extract
    train_df, test_df, article_df = load_data()

    # import pickle
    # pickle.dump(hm_data, open('hm_data.p', 'wb'))
    # hm_data = pickle.load(open('hm_data.p', 'rb'))

    # Preprocess
    preprocessed_hm_data = preprocess(train_df, test_df, article_df, batch_size=config.batch_size)

    # Train
    return run_training(preprocessed_hm_data, config)


if __name__ == '__main__':
    all_results = []
    for emb_dim in [128, 64]:
        for learning_rate in [0.01, 0.05, 0.1]:
            for batch_size in [256, 512, 1024]:
                print('')
                print(f'Running with emb_dim: {emb_dim}, learning_rate: {learning_rate}, batch_size: {batch_size}')
                print('')
                config = Config(embedding_dimension=emb_dim,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                nb_epochs=4)
                history = run_all(config)
                results = history.history
                results.update(config.to_json())

                all_results.append(results)

    import pickle
    pickle.dump(all_results, open('all_results.p', 'wb'))
