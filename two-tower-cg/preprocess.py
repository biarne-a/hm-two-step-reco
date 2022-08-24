import pandas as pd
import tensorflow as tf
from typing import Dict


class PreprocessedHmData:
    def __init__(self,
                 train_ds: tf.data.Dataset,
                 nb_train_obs: int,
                 test_ds: tf.data.Dataset,
                 nb_test_obs: int,
                 all_articles: tf.Tensor,
                 label_probs: tf.lookup.StaticHashTable,
                 article_vocab_size: int,
                 customer_vocab_size: int):
        self.train_ds = train_ds
        self.nb_train_obs = nb_train_obs
        self.test_ds = test_ds
        self.nb_test_obs = nb_test_obs
        self.all_articles = all_articles
        self.label_probs = label_probs
        self.article_vocab_size = article_vocab_size
        self.customer_vocab_size = customer_vocab_size


def split_data(transactions_df):
    trans_date = transactions_df['t_dat']
    train_df = transactions_df[(trans_date >= '2019-09-20') & (trans_date <= '2020-08-20')]
    test_df = transactions_df[trans_date >= '2020-08-20']
    return train_df, test_df


def perform_string_lookups(inputs: Dict[str, tf.Tensor],
                           article_lookup: tf.keras.layers.StringLookup,
                           customer_lookup: tf.keras.layers.StringLookup) -> Dict[str, tf.Tensor]:
    return {
        'article_id': article_lookup(inputs['article_id']),
        'customer_id': customer_lookup(inputs['customer_id'])
    }


def get_label_probs(train_df: pd.DataFrame,
                    article_lookup: tf.keras.layers.StringLookup) -> tf.lookup.StaticHashTable:
    article_counts_dict = train_df.groupby('article_id')['article_id'].count().to_dict()
    nb_transactions = train_df.shape[0]
    keys = list(article_counts_dict.keys())
    values = [count / nb_transactions for count in article_counts_dict.values()]

    keys = tf.constant(keys, dtype=tf.string)
    keys = article_lookup(keys)
    values = tf.constant(values, dtype=tf.float32)

    return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values),
                                     default_value=0.0)


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, article_df: pd.DataFrame, batch_size: int) -> PreprocessedHmData:
    # train_df, test_df = split_data(data.transactions_df)
    nb_train_obs = train_df.shape[0]
    nb_test_obs = test_df.shape[0]

    unique_article_ids = train_df.article_id.unique()
    article_lookup = tf.keras.layers.StringLookup(vocabulary=unique_article_ids)
    article_vocab_size = len(unique_article_ids) + 1
    print(f'article vocab size = {article_vocab_size}')

    unique_customer_ids = train_df.customer_id.unique()
    customer_lookup = tf.keras.layers.StringLookup(vocabulary=unique_customer_ids)
    customer_vocab_size = len(unique_customer_ids) + 1
    print(f'customer vocab size = {article_vocab_size}')

    train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df[['customer_id', 'article_id']])) \
        .shuffle(100_000) \
        .batch(batch_size) \
        .map(lambda inputs: perform_string_lookups(inputs, article_lookup, customer_lookup)) \
        .repeat()
    test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df[['customer_id', 'article_id']])) \
        .batch(batch_size) \
        .map(lambda inputs: perform_string_lookups(inputs, article_lookup, customer_lookup)) \
        .repeat()
    all_articles_with_oov = [article_lookup.oov_token] + list(article_lookup.input_vocabulary)
    articles_ds = tf.data.Dataset.from_tensor_slices(all_articles_with_oov) \
        .batch(article_vocab_size) \
        .map(article_lookup)
    all_articles = next(iter(articles_ds))
    label_probs = get_label_probs(train_df, article_lookup)

    return PreprocessedHmData(train_ds,
                              nb_train_obs,
                              test_ds,
                              nb_test_obs,
                              all_articles,
                              label_probs,
                              article_vocab_size,
                              customer_vocab_size)
