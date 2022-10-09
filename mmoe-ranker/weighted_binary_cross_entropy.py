import tensorflow as tf


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self,
                 negative_class_weight: float,
                 positve_class_weight: float):
        super().__init__()
        self._negative_class_weight = tf.constant(negative_class_weight)
        self._positve_class_weight = tf.constant(positve_class_weight)
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                       reduction=tf.keras.losses.Reduction.NONE)

    def call(self, true_labels, predictions):
        batch_loss_values = self._bce(true_labels, predictions)
        batch_loss_values = tf.where(true_labels == tf.constant(0.0),
                                     batch_loss_values * self._negative_class_weight,
                                     batch_loss_values * self._positve_class_weight)
        return tf.reduce_sum(batch_loss_values)
