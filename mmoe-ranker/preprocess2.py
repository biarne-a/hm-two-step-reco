import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, Tuple, List

from features import Features
from load_data2 import HmData


class PreprocessedHmData:
    def __init__(self,
                 train_ds: tf.data.Dataset,
                 nb_train_obs: int,
                 test_ds: tf.data.Dataset,
                 nb_test_obs: int,
                 lookups: Dict[str, tf.keras.layers.StringLookup],
                 normalization_layers: Dict[str, tf.keras.layers.Normalization]):
        self.train_ds = train_ds
        self.nb_train_obs = nb_train_obs
        self.test_ds = test_ds
        self.nb_test_obs = nb_test_obs
        self.lookups = lookups
        self.normalization_layers = normalization_layers


def prepare_batch(inputs: Dict[str, tf.Tensor],
                  lookups: Dict[str, tf.keras.layers.StringLookup]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    batch_inputs = {}
    for key, value in inputs.items():
        if key in lookups:
            batch_inputs[key] = lookups[key](value)
        else:
            batch_inputs[key] = value
    label = inputs[Features.LABEL]
    return batch_inputs, label


def build_lookups(train_df: pd.DataFrame) -> Dict[str, tf.keras.layers.StringLookup]:
    print('Building lookups')
    lookups = {}
    for categ_variable in Features.ALL_CATEG_FEATURES:
        unique_values = train_df[categ_variable].unique()
        lookups[categ_variable] = tf.keras.layers.StringLookup(vocabulary=unique_values)
    return lookups


def build_normalization_layers(data: HmData) -> Dict[str, tf.keras.layers.Normalization]:
    print('Building normalization layers')
    normalization_layers = {}
    for key in tqdm(data.engineered_columns + Features.ALL_CONTI_FEATURES):
        feature = data.train_df[key]
        normalization_layer = tf.keras.layers.Normalization(axis=None, mean=np.mean(feature), variance=np.var(feature))
        normalization_layers[key] = normalization_layer
    return normalization_layers


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


def generative_negatives(customer_ids: List[str],
                         engineered_customer_columns: List[str],
                         engineered_article_columns: List[str],
                         neg_by_cust_func,
                         customer_categ_dicts: Dict[str, Dict[str, str]],
                         customer_engineered_dicts: Dict[str, Dict[str, float]],
                         article_categ_dicts: Dict[str, Dict[str, str]],
                         article_engineered_dicts: Dict[str, Dict[str, float]]):
    for customer_id in customer_ids:
        negatives = neg_by_cust_func(customer_id)
        for j in range(len(negatives)):
            obs = {}
            neg_article_id = negatives[j]
            # We set the customer features for current customer
            for key in Features.CUSTOMER_FEATURES:
                obs[key] = customer_categ_dicts[customer_id][key]
            for key in engineered_customer_columns:
                obs[key] = customer_engineered_dicts[customer_id][key]
            # We set the article features for sampled negative article
            for key in Features.ARTICLE_CATEG_FEATURES:
                obs[key] = article_categ_dicts[neg_article_id][key]
            for key in engineered_article_columns:
                obs[key] = article_engineered_dicts[neg_article_id][key]
            obs[Features.LABEL] = 0.0
            yield obs


def preprocess(data: HmData, batch_size: int) -> PreprocessedHmData:
    print('Preprocessing')
    lookups = build_lookups(data.train_df)

    normalization_layers = build_normalization_layers(data)

    total_count = len(data.train_df)
    article_ids = []
    article_probs = []
    for article_id, count in data.all_articles_counts.items():
        article_ids.append(article_id)
        article_probs.append(count / total_count)

    nb_negatives = 10
    nb_transac_by_cust = data.train_df.groupby('customer_id').size().to_dict()
    def neg_by_cust_func(customer_id):
        nb_transac_for_cust = nb_transac_by_cust[customer_id]
        nb_negatives_for_cust = nb_negatives * nb_transac_for_cust
        return np.random.choice(article_ids, size=nb_negatives_for_cust, p=article_probs)

    customer_categ_dicts = data.customer_df.set_index('customer_id', drop=False).to_dict(orient='index')
    customer_engineered_dicts = data.engineered_customer_features.set_index('customer_id', drop=True).to_dict(orient='index')
    article_categ_dicts = data.article_df.set_index('article_id', drop=False).to_dict(orient='index')
    article_engineered_dicts = data.engineered_article_features.set_index('article_id', drop=True).to_dict(orient='index')

    def gen_negatives():
        return generative_negatives(data.train_df.customer_id.to_list(),
                                    data.engineered_customer_columns,
                                    data.engineered_article_columns,
                                    neg_by_cust_func,
                                    customer_categ_dicts,
                                    customer_engineered_dicts,
                                    article_categ_dicts,
                                    article_engineered_dicts)

    all_variables = Features.ALL_VARIABLES + data.engineered_columns
    data.train_df[Features.LABEL] = 1.0
    pos_train_ds = tf.data.Dataset.from_tensor_slices(dict(data.train_df[all_variables]))
    neg_train_ds = tf.data.Dataset.from_generator(gen_negatives, output_signature=pos_train_ds.element_spec)
    train_ds = pos_train_ds.concatenate(neg_train_ds) \
        .shuffle(100_000) \
        .batch(batch_size) \
        .map(lambda inputs: prepare_batch(inputs, lookups)) \
        .repeat()
    pos_test_ds = tf.data.Dataset.from_tensor_slices(dict(data.test_df[all_variables]))
    neg_test_ds = tf.data.Dataset.from_generator(gen_negatives, output_signature=pos_test_ds.element_spec)
    test_ds = pos_test_ds.concatenate(neg_test_ds) \
        .batch(batch_size) \
        .map(lambda inputs: prepare_batch(inputs, lookups)) \
        .repeat()

    nb_train_obs = data.train_df.shape[0] + nb_negatives * data.train_df.shape[0]
    nb_test_obs = data.test_df.shape[0] + nb_negatives * data.test_df.shape[0]

    print(f'nb_train_obs: {nb_train_obs}')
    print(f'nb_test_obs: {nb_test_obs}')

    return PreprocessedHmData(train_ds,
                              nb_train_obs,
                              test_ds,
                              nb_test_obs,
                              lookups,
                              normalization_layers)
