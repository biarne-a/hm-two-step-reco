import tensorflow as tf
from config import Config
from preprocess import PreprocessedHmData

from basic_ranker import BasicRanker


def run_training(data: PreprocessedHmData, config: Config):
    model = BasicRanker(data)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.AUC(curve='PR')],
                  run_eagerly=False)

    weight_for_0 = 1.0
    weight_for_1 = 10.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    history = model.fit(x=data.train_ds,
                        epochs=config.nb_epochs,
                        steps_per_epoch=data.nb_train_obs // config.batch_size,
                        validation_data=data.test_ds,
                        validation_steps=data.nb_test_obs // config.batch_size,
                        class_weight=class_weight,
                        verbose=1)
    return model, history

