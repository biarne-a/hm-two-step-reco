import math
import tensorflow as tf
from tensorflow import keras

from features import Features
from preprocess import PreprocessedHmData


class BasicRanker(keras.models.Model):
    def __init__(self, data: PreprocessedHmData):
        super().__init__()
        self._emb_layers = {key: self._build_embedding_layer(lkp) for key, lkp in data.lookups.items()}
        self._normalization_layers = data.normalization_layers
        self._dense1 = keras.layers.Dense(256, activation='relu')
        self._dense2 = keras.layers.Dense(128, activation='relu')
        self._dense3 = keras.layers.Dense(64, activation='relu')
        self._dense4 = keras.layers.Dense(32, activation='relu')
        self._dense_layers = [self._dense1, self._dense2, self._dense3, self._dense4, self._dense5]
        self._main_label_dense = keras.layers.Dense(1, activation='sigmoid', name="output1")
        nb_labels2 = len(data.lookups[Features.LABEL2].input_vocabulary)
        self._label2_dense = keras.layers.Dense(nb_labels2, activation='softmax', name="output2")

    def _build_embedding_layer(self, lookup: keras.layers.StringLookup):
        vocab_size = len(lookup.input_vocabulary) + 1
        emb_dim = int(3 * math.log(vocab_size, 2))
        return keras.layers.Embedding(vocab_size, emb_dim)

    def call(self, inputs, training):
        vars_to_concat = []
        # Embed categorical variables
        for key, embedding_layer in self._emb_layers.items():
            embedded_var = embedding_layer(inputs[key])
            vars_to_concat.append(embedded_var)
        # Normalize numerical features
        for key, norm_layer in self._normalization_layers.items():
            normalized_var = norm_layer(inputs[key])
            vars_to_concat.append(tf.expand_dims(normalized_var, axis=1))
        concatanated_vars = tf.concat(vars_to_concat, axis=1)
        dense_outputs = concatanated_vars
        for dense in self._dense_layers:
            dense_outputs = dense(dense_outputs)
        main_label_output = self._main_label_dense(dense_outputs)
        label2_output = self._label2_dense(dense_outputs)
        return main_label_output, label2_output
