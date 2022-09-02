import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from typing import Dict, Tuple, List

from features import Features


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


def build_lookups(train_df) -> Dict[str, tf.keras.layers.StringLookup]:
    print('Building lookups')
    lookups = {}
    for categ_variable in Features.ALL_CATEG_FEATURES:
        unique_values = train_df[categ_variable].unique()
        lookups[categ_variable] = tf.keras.layers.StringLookup(vocabulary=unique_values)
    return lookups


def build_normalization_layers(engineered_features: List[str],
                               train_df: pd.DataFrame) -> Dict[str, tf.keras.layers.Normalization]:
    print('Building normalization layers')
    normalization_layers = {}
    for key in tqdm(engineered_features + Features.ALL_CONTI_FEATURES):
        feature = train_df[key]
        normalization_layer = tf.keras.layers.Normalization(axis=None, mean=np.mean(feature), variance=np.var(feature))
        normalization_layers[key] = normalization_layer
    return normalization_layers


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               batch_size: int,
               engineered_features: List[str]) -> PreprocessedHmData:
    print('Preprocessing')
    nb_train_obs = train_df.shape[0]
    nb_test_obs = test_df.shape[0]

    lookups = build_lookups(train_df)

    normalization_layers = build_normalization_layers(engineered_features, train_df)

    all_variables = Features.ALL_VARIABLES + engineered_features
    train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df[all_variables])) \
        .shuffle(100_000) \
        .batch(batch_size) \
        .map(lambda inputs: prepare_batch(inputs, lookups)) \
        .repeat()
    test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df[all_variables])) \
        .batch(batch_size) \
        .map(lambda inputs: prepare_batch(inputs, lookups)) \
        .repeat()

    return PreprocessedHmData(train_ds,
                              nb_train_obs,
                              test_ds,
                              nb_test_obs,
                              lookups,
                              normalization_layers)
