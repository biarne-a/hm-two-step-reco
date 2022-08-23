import math
from typing import Dict
import tensorflow as tf
from tensorflow import keras


class MultipleEmbeddingsLayer(keras.layers.Layer):
    def __init__(self, lookups: Dict[str, tf.keras.layers.StringLookup]):
        super().__init__()
        self._all_embeddings = {}
        for categ_variable in lookups.keys():
            lookup = lookups[categ_variable]
            vocab_size = len(lookup.input_vocabulary) + 1
            cat_var_emb_dim = int(3 * math.log(vocab_size, 2))
            embedding_layer = tf.keras.layers.Embedding(vocab_size, cat_var_emb_dim)
            self._all_embeddings[categ_variable] = embedding_layer

    def call(self, inputs):
        all_embeddings = []
        for variable, embedding_layer in self._all_embeddings.items():
            embeddings = embedding_layer(inputs[variable])
            all_embeddings.append(embeddings)
        return tf.concat(all_embeddings, axis=1)
