import tensorflow as tf
from tensorflow import keras

from preprocess import PreprocessedHmData
from custom_cross_entropy_loss import CustomCrossEntropyLoss


class Basic2TowerModel(keras.models.Model):
    def __init__(self,
                 customer_model: keras.models.Model,
                 article_model: keras.models.Model,
                 data: PreprocessedHmData):
        super().__init__()
        self._article_model = article_model
        self._customer_model = customer_model
        self._all_articles = data.all_articles
        self._custom_loss = CustomCrossEntropyLoss(label_probs=data.label_probs_hash_table)

    def call(self, inputs, training=False):
        customer_embeddings = self._customer_model(inputs)

        if training:
            article_embeddings = self._article_model(inputs)
        else:
            article_embeddings = self._article_model(self._all_articles)

        return tf.matmul(customer_embeddings, tf.transpose(article_embeddings))

    def train_step(self, inputs):
        # Forward pass
        with tf.GradientTape() as tape:
            logits = self(inputs, training=True)
            loss_val = self._custom_loss(inputs['article_id'], logits, training=True)

        # Backward pass
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        return {'loss': loss_val}

    def test_step(self, inputs):
        # Forward pass
        logits = self(inputs, training=False)
        loss_val = self._custom_loss(inputs['article_id'], logits, training=False)

        # Compute metrics
        # We add one to the output indices because everything is shifted because of the OOV token
        top_indices = tf.math.top_k(logits, k=1000).indices + 1
        metric_results = self.compute_metrics(x=None, y=inputs['article_id'], y_pred=top_indices, sample_weight=None)

        return {'loss': loss_val, **metric_results}
