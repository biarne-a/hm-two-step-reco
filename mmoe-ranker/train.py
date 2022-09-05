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

    # model.fit(x=data.train_ds, epochs=1, steps_per_epoch=1, verbose=1)
    # print(model.summary())

    # neg = 7906360
    # pos = 93640
    # total = neg + pos
    # weight_for_0 = (1 / neg) * (total / 2.0)
    # weight_for_1 = (1 / pos) * (total / 2.0)
    weight_for_0 = 1
    weight_for_1 = 10

    class_weight = {0: weight_for_0, 1: weight_for_1}

    return model.fit(x=data.train_ds,
                     epochs=config.nb_epochs,
                     # steps_per_epoch=data.nb_train_obs // config.batch_size,
                     steps_per_epoch=4000,
                     validation_data=data.test_ds,
                     validation_steps=data.nb_test_obs // config.batch_size,
                     class_weight=class_weight,
                     verbose=1)
