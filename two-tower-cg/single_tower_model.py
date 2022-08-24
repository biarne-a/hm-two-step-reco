import math
from typing import Dict
import tensorflow as tf
from tensorflow import keras


class SingleTowerModel(keras.models.Model):
    def __init__(self,
                lookups: Dict[str, tf.keras.layers.StringLookup],
                embedding_dimension: int):
        super().__init__()
        self._all_embeddings = {}
        for categ_variable in lookups.keys():
            lookup = lookups[categ_variable]
            vocab_size = len(lookup.input_vocabulary) + 1
            if categ_variable == 'article_id' or categ_variable == 'customer_id':
                cat_var_emb_dim = 128
            else:
                cat_var_emb_dim = int(3 * math.log(vocab_size, 2))
            embedding_layer = tf.keras.layers.Embedding(vocab_size, cat_var_emb_dim)
            self._all_embeddings[categ_variable] = embedding_layer

        self._dense1 = tf.keras.layers.Dense(256, activation='relu')
        self._dense2 = tf.keras.layers.Dense(embedding_dimension, activation='relu')

    def call(self, inputs):
        all_embeddings = []
        for variable, embedding_layer in self._all_embeddings.items():
            embeddings = embedding_layer(inputs[variable])
            all_embeddings.append(embeddings)
        all_embeddings = tf.concat(all_embeddings, axis=1)
        outputs = self._dense1(all_embeddings)
        return self._dense2(outputs)
