import tensorflow as tf
from typing import Dict
from tensorflow import keras

from config import Config, Variables
from custom_recall import CustomRecall
from preprocess import PreprocessedHmData
from basic_2_tower_model import Basic2TowerModel
from mutliple_embeddings_layer import MultipleEmbeddingsLayer


def build_tower_sub_model(vocab_size: int, embedding_dimension: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dimension),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(embedding_dimension, activation='relu')
    ])


def build_complex_sub_model(lookups: Dict[str, tf.keras.layers.StringLookup],
                            embedding_dimension: int):
    return tf.keras.Sequential([
        MultipleEmbeddingsLayer(lookups),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(embedding_dimension, activation='relu')
    ])


def get_callbacks():
    return [keras.callbacks.TensorBoard(log_dir='logs', update_freq=100)]


def run_training(data: PreprocessedHmData,
                 config: Config):
    article_lookups = {key: lkp for key, lkp in data.lookups.items() if key in Variables.ARTICLE_CATEG_VARIABLES}
    article_model = build_complex_sub_model(article_lookups, config.embedding_dimension)
    customer_lookups = {key: lkp for key, lkp in data.lookups.items() if key in Variables.CUSTOMER_CATEG_VARIABLES}
    customer_model = build_complex_sub_model(customer_lookups, config.embedding_dimension)

    model = Basic2TowerModel(customer_model, article_model, data)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate),
                  metrics=[CustomRecall(k=100), CustomRecall(k=500), CustomRecall(k=1000)],
                  run_eagerly=True)

    return model.fit(x=data.train_ds,
                     epochs=config.nb_epochs,
                     # steps_per_epoch=data.nb_train_obs // config.batch_size,
                     steps_per_epoch=3,
                     validation_data=data.test_ds,
                     # validation_steps=data.nb_test_obs // config.batch_size,
                     validation_steps=3,
                     callbacks=get_callbacks(),
                     verbose=1)
