import tensorflow as tf
from config import Config
from preprocess import PreprocessedHmData

from basic_ranker import BasicRanker


def run_training(data: PreprocessedHmData, config: Config):
    model = BasicRanker(data)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy()],
                  run_eagerly=False)

    return model.fit(x=data.train_ds,
                     epochs=config.nb_epochs,
                     steps_per_epoch=data.nb_train_obs // config.batch_size,
                     # steps_per_epoch=4000,
                     validation_data=data.test_ds,
                     validation_steps=data.nb_test_obs // config.batch_size,
                     verbose=1)
