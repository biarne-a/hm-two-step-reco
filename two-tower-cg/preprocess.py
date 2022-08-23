import pandas as pd
import tensorflow as tf
from typing import Dict

from config import Variables


class PreprocessedHmData:
    def __init__(self,
                 train_ds: tf.data.Dataset,
                 nb_train_obs: int,
                 test_ds: tf.data.Dataset,
                 nb_test_obs: int,
                 lookups: Dict[str, tf.keras.layers.StringLookup],
                 all_articles: Dict[str, tf.Tensor],
                 label_probs_hash_table: tf.lookup.StaticHashTable,
                 full_article_probs: tf.Tensor):
        self.train_ds = train_ds
        self.nb_train_obs = nb_train_obs
        self.test_ds = test_ds
        self.nb_test_obs = nb_test_obs
        self.lookups = lookups
        self.all_articles = all_articles
        self.label_probs_hash_table = label_probs_hash_table
        self.full_article_probs = full_article_probs


def perform_string_lookups(inputs: Dict[str, tf.Tensor],
                           lookups: Dict[str, tf.keras.layers.StringLookup]) -> Dict[str, tf.Tensor]:
    return {key: lkp(inputs[key]) for key, lkp in lookups.items()}


def get_label_probs_hash_table(train_df: pd.DataFrame,
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


def build_lookups(train_df) -> Dict[str, tf.keras.layers.StringLookup]:
    lookups = {}
    for categ_variable in Variables.ALL_CATEG_VARIABLES:
        unique_values = train_df[categ_variable].unique()
        lookups[categ_variable] = tf.keras.layers.StringLookup(vocabulary=unique_values)
    return lookups


def preprocess(train_df, test_df, article_df, batch_size) -> PreprocessedHmData:
    nb_train_obs = train_df.shape[0]
    nb_test_obs = test_df.shape[0]

    lookups = build_lookups(train_df)

    train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df[Variables.ALL_CATEG_VARIABLES])) \
        .shuffle(100_000) \
        .batch(batch_size) \
        .map(lambda inputs: perform_string_lookups(inputs, lookups)) \
        .repeat()
    test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df[Variables.ALL_CATEG_VARIABLES])) \
        .batch(batch_size) \
        .map(lambda inputs: perform_string_lookups(inputs, lookups)) \
        .repeat()
    article_lookups = {key: lkp for key, lkp in lookups.items() if key in Variables.ARTICLE_CATEG_VARIABLES}
    article_ds = tf.data.Dataset.from_tensor_slices(dict(article_df[Variables.ARTICLE_CATEG_VARIABLES])) \
        .batch(len(article_df)) \
        .map(lambda inputs: perform_string_lookups(inputs, article_lookups))
    all_articles = next(iter(article_ds))

    article_lookup = lookups['article_id']
    all_article_ids = article_lookup([article_lookup.oov_token] + list(article_lookup.input_vocabulary))
    label_probs_hash_table = get_label_probs_hash_table(train_df, article_lookup)
    full_article_probs = label_probs_hash_table.lookup(all_article_ids)

    return PreprocessedHmData(train_ds,
                              nb_train_obs,
                              test_ds,
                              nb_test_obs,
                              lookups,
                              all_articles,
                              label_probs_hash_table,
                              full_article_probs)
