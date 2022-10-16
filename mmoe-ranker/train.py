import tensorflow as tf
import tensorflow_ranking as tfr

from config import Config
from basic_ranker import BasicRanker
from preprocess import PreprocessedHmData
from weighted_binary_cross_entropy import WeightedBinaryCrossEntropy


def run_training(data: PreprocessedHmData, config: Config):
    model = BasicRanker(data)
    weighted_bce_loss = WeightedBinaryCrossEntropy(negative_class_weight=1.0, positve_class_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate),
                  # loss={
                  #     'output1': weighted_bce_loss,
                  #     'output2': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  # },
                  # metrics={
                  #     'output1': tf.keras.metrics.AUC(curve='PR'),
                  #     'output2': [tfr.keras.metrics.PrecisionMetric(topn=1),
                  #                 tfr.keras.metrics.RecallMetric(topn=1)]
                  # },
                  loss=weighted_bce_loss,
                  metrics=[tf.keras.metrics.AUC(curve='PR'),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision()],
                  run_eagerly=False)

    history = model.fit(x=data.train_ds,
                        epochs=config.nb_epochs,
                        steps_per_epoch=data.nb_train_obs // config.batch_size,
                        validation_data=data.test_ds,
                        validation_steps=data.nb_test_obs // config.batch_size,
                        verbose=1)
    return model, history
