import tensorflow as tf


class CustomRecall(tf.keras.metrics.Metric):
    def __init__(self,
                 k: int,
                 name='recall'):
        super().__init__(name=f'{name}_at_{k}')
        self._cumulative_recall = tf.Variable(0.)
        self._sum_weights = tf.Variable(0.)
        self._k = k

    def _recall_at_k(self, true_indices, top_indices):
        batch_size = tf.shape(top_indices)[0]
        top_indices = tf.slice(top_indices, [0, 0], [batch_size, self._k])
        repeated_correct_item_ids = tf.repeat(true_indices, repeats=[self._k], axis=1)
        recall = tf.cast(repeated_correct_item_ids == top_indices, dtype=tf.float32)

        return tf.reduce_sum(recall, axis=1,  keepdims=True)

    def update_state(self, true_labels, top_indices, sample_weight=None):
        """Accumulates true positive and false negative statistics.
        Args:
          true_labels: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted logits.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        """
        true_labels = tf.cast(true_labels, dtype=tf.int32)
        recall_k = self._recall_at_k(true_labels, top_indices)
        if sample_weight is not None:
            recall_k = recall_k * sample_weight
            self._sum_weights.assign_add(tf.reduce_sum(sample_weight))
        else:
            # give equal weight to all samples
            self._sum_weights.assign_add(tf.cast(tf.shape(recall_k)[0], dtype=tf.float32))

        self._cumulative_recall.assign_add(tf.reduce_sum(recall_k))

    def result(self):
        return tf.math.divide_no_nan(
            self._cumulative_recall,
            self._sum_weights
        )

    def reset_state(self):
        self._cumulative_recall.assign(0.)
        self._sum_weights.assign(0.)
