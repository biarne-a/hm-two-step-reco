import tensorflow as tf
import tensorflow_ranking as tfr

from config import Config
from basic_ranker import BasicRanker
from preprocess import PreprocessedHmData


def run_training(data: PreprocessedHmData, config: Config):
    model = BasicRanker(data)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate),
                  loss={
                      'output1': tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      'output2': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  },
                  metrics={
                      'output1': tf.keras.metrics.AUC(curve='PR'),
                      'output2': [tfr.keras.metrics.PrecisionMetric(topn=1),
                                  tfr.keras.metrics.RecallMetric(topn=1)]
                  },
                  run_eagerly=False)

    weight_for_0 = 1.0
    weight_for_1 = 10.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    history = model.fit(x=data.train_ds,
                        epochs=config.nb_epochs,
                        # steps_per_epoch=data.nb_train_obs // config.batch_size,
                        steps_per_epoch=100_000,
                        validation_data=data.test_ds,
                        validation_steps=data.nb_test_obs // config.batch_size,
                        class_weight=class_weight,
                        verbose=1)
    return model, history
