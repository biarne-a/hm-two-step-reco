import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, Tuple, List

from features import Features
from load_data import HmData


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


def save_dataset(ds, x_filename, y_filename):
    print('Saving dataset')
    from collections import defaultdict
    all_x = defaultdict(list)
    all_y = defaultdict(list)
    for x, y in iter(ds):
        for x_key, x_values in x.items():
            all_x[x_key].append(x_values.numpy())
        # for y_key, y_values in y.items():
        all_y['output1'].append(y.numpy())

    x = {}
    y = {}
    for x_key, x_list_values in all_x.items():
        x[x_key] = np.concatenate(x_list_values)
    for y_key, y_list_values in all_y.items():
        y[y_key] = np.concatenate(y_list_values)

    import pickle
    pickle.dump(x, open(x_filename, 'wb'))
    pickle.dump(y, open(y_filename, 'wb'))


def generate_test_dataset(add_neg_article_info, all_cust_vars, all_variables, batch_size, data, nb_negatives,
                          lookups, one_hot_encoding_layer):
    print('Generating test dataset')
    pos_test_ds = tf.data.Dataset.from_tensor_slices(dict(data.test_df[all_variables]))
    neg_test_ds = tf.data.Dataset.from_tensor_slices(dict(data.test_df[all_cust_vars])) \
        .repeat(count=nb_negatives) \
        .map(add_neg_article_info, num_parallel_calls=tf.data.AUTOTUNE)
    choice_dataset = tf.data.Dataset.from_tensors(tf.constant(0, dtype=tf.int64)) \
        .concatenate(tf.data.Dataset.from_tensors(tf.constant(1, dtype=tf.int64)).repeat(nb_negatives)) \
        .repeat(len(data.test_df))
    return tf.data.Dataset.choose_from_datasets([pos_test_ds, neg_test_ds], choice_dataset) \
        .batch(batch_size) \
        .map(lambda inputs: prepare_batch(inputs, lookups, one_hot_encoding_layer))


def prepare_batch(
    inputs: Dict[str, tf.Tensor],
    lookups: Dict[str, tf.keras.layers.StringLookup],
    one_hot_encoding_layer: tf.keras.layers.CategoryEncoding,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    batch_inputs = {}
    for key, value in inputs.items():
        if key in lookups:
            batch_inputs[key] = lookups[key](value)
        else:
            batch_inputs[key] = value
    label1 = inputs[Features.LABEL1]

    # label2_lookup = lookups[Features.LABEL2]
    # label2_indice = label2_lookup(inputs[Features.LABEL2])
    # label2_one_hot = one_hot_encoding_layer(label2_indice)

    return batch_inputs, label1

    # outputs = {
    #     'output1': label1,
    #     'output2': label2_one_hot,
    # }
    # return batch_inputs, outputs


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
            obs[Features.LABEL1] = 0.0
            yield obs


def build_hash_tables(article_dicts: Dict[str, Dict[str, str]],
                      value_type,
                      default_value) -> Dict[str, tf.lookup.StaticHashTable]:
    hash_tables = {}
    for feature, feature_values in article_dicts.items():
        keys = list(feature_values.keys())
        values = list(feature_values.values())
        keys = tf.constant(keys, dtype=tf.string)
        values = tf.constant(values, dtype=value_type)
        hash_tables[feature] = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values),
                                                         default_value=default_value)
    return hash_tables


def preprocess(data: HmData, batch_size: int) -> PreprocessedHmData:
    print('Preprocessing')
    lookups = build_lookups(data.train_df)

    normalization_layers = build_normalization_layers(data)

    total_count = len(data.train_df)
    article_ids = lookups['article_id'].input_vocabulary
    article_probs = []
    for article_id in article_ids:
        count = data.all_articles_counts[article_id]
        article_probs.append(count / total_count)

    nb_negatives = 1
    nb_train_obs = data.train_df.shape[0] + nb_negatives * data.train_df.shape[0]
    nb_test_obs = data.test_df.shape[0] + nb_negatives * data.test_df.shape[0]

    print(f'nb_train_obs: {nb_train_obs}')
    print(f'nb_test_obs: {nb_test_obs}')

    article_categ_dicts = data.article_df.set_index('article_id', drop=False).to_dict()
    article_categ_hash_tables = build_hash_tables(article_categ_dicts, value_type=tf.string, default_value="")
    article_engineered_dicts = data.engineered_article_features.set_index('article_id', drop=True).to_dict()
    article_engineered_hash_tables = build_hash_tables(article_engineered_dicts, value_type=tf.float64,
                                                       default_value=0.0)

    all_neg_article_ids = tf.constant(np.random.choice(article_ids, size=nb_train_obs, p=article_probs),
                                      dtype=tf.string)



    def add_neg_article_info(inputs):
        neg_article_id = tf.gather(all_neg_article_ids, inputs['idx'])
        inputs['article_id'] = neg_article_id
        for feature, hash_table in article_categ_hash_tables.items():
            inputs[feature] = hash_table.lookup(neg_article_id)
        for feature, hash_table in article_engineered_hash_tables.items():
            inputs[feature] = hash_table.lookup(neg_article_id)
        inputs[Features.LABEL1] = tf.constant(0.0, dtype=tf.float64)
        for key, value in inputs.items():
            inputs[key] = tf.reshape(value, shape=())
        return inputs

    for key in data.train_df.columns:
        if '_nb_' in key:
            data.train_df[key] = data.train_df[key].astype(np.float64)

    label2_lookup = lookups[Features.LABEL2]
    num_label2 = len(label2_lookup.input_vocabulary) + 1
    one_hot_encoding_layer = tf.keras.layers.CategoryEncoding(num_tokens=num_label2, output_mode="one_hot")

    all_variables = Features.ALL_VARIABLES + data.engineered_columns + ['idx']
    data.train_df['idx'] = np.arange(len(data.train_df))
    data.test_df['idx'] = np.arange(len(data.test_df))
    pos_train_ds = tf.data.Dataset.from_tensor_slices(dict(data.train_df[all_variables]))
    all_cust_vars = Features.CUSTOMER_FEATURES + data.engineered_customer_columns + ['idx']
    neg_train_ds = tf.data.Dataset.from_tensor_slices(dict(data.train_df[all_cust_vars])) \
        .repeat(count=nb_negatives) \
        .map(add_neg_article_info, num_parallel_calls=tf.data.AUTOTUNE)
    choice_dataset = tf.data.Dataset.from_tensors(tf.constant(0, dtype=tf.int64)) \
        .concatenate(tf.data.Dataset.from_tensors(tf.constant(1, dtype=tf.int64)).repeat(nb_negatives)) \
        .repeat(len(data.train_df))
    train_ds = tf.data.Dataset.choose_from_datasets([pos_train_ds, neg_train_ds], choice_dataset) \
        .shuffle(100_000) \
        .batch(batch_size) \
        .map(lambda inputs: prepare_batch(inputs, lookups, one_hot_encoding_layer))

    if os.path.exists('test_x.p') and os.path.exists('test_y.p'):
        test_x = pickle.load(open('test_x.p', 'rb'))
        test_y = pickle.load(open('test_y.p', 'rb'))

        test_ds_x = tf.data.Dataset.from_tensor_slices(test_x)
        test_ds_y = tf.data.Dataset.from_tensor_slices(test_y['output1'])
        test_ds = tf.data.Dataset.zip((test_ds_x, test_ds_y)).batch(batch_size)
    else:
        test_ds = generate_test_dataset(add_neg_article_info, all_cust_vars, all_variables, batch_size, data, nb_negatives,
                                        lookups, one_hot_encoding_layer)
        save_dataset(test_ds, 'test_x.p', 'test_y.p')

    save_dataset(train_ds, 'train_x.p', 'train_y.p')

    return PreprocessedHmData(train_ds.repeat(),
                              nb_train_obs,
                              test_ds.repeat(),
                              nb_test_obs,
                              lookups,
                              normalization_layers)
