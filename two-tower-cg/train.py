import tensorflow as tf
from tensorflow import keras

from preprocess import PreprocessedHmData
from basic_2_tower_model import Basic2TowerModel
from custom_recall import CustomRecall
from config import Config


def build_tower_sub_model(vocab_size: int, embedding_dimension: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dimension),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(embedding_dimension, activation='relu')
    ])


def get_callbacks():
    return [keras.callbacks.TensorBoard(log_dir='logs', update_freq=100)]


def run_training(data: PreprocessedHmData,
                 config: Config):
    article_model = build_tower_sub_model(data.article_vocab_size, config.embedding_dimension)
    customer_model = build_tower_sub_model(data.customer_vocab_size, config.embedding_dimension)

    model = Basic2TowerModel(customer_model, article_model, data)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate),
                  metrics=[CustomRecall(k=100), CustomRecall(k=500), CustomRecall(k=1000)],
                  run_eagerly=True)

    model.fit(x=data.train_ds,
              epochs=config.nb_epochs,
              steps_per_epoch=data.nb_train_obs // config.batch_size,
              validation_data=data.test_ds,
              validation_steps=data.nb_test_obs // config.batch_size,
              callbacks=get_callbacks(),
              verbose=1)
