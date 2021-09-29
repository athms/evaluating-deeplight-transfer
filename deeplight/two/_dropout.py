#!/usr/bin/python
import tensorflow as tf
from modules import Module


class Dropout(Module):
    '''
    Dropout Layer
    '''
    def __init__(self, keep_prob=tf.constant(1.0), noise_shape=None, name='dropout', batch_size=None):
        self.name = name
        Module.__init__(self)

        self.keep_prob = keep_prob
        self.noise_shape = noise_shape
        self.batch_size = batch_size

    def forward(self, input_tensor):
        """Forward pass"""
        self.input_tensor = input_tensor
        inp_shape = self.input_tensor.get_shape().as_list()

        with tf.name_scope(self.name):

            def dropout_check_false():
                return tf.cast(tf.constant(1.0), dtype=tf.float32)

            def dropout_check_true():
                return tf.cast(tf.multiply(self.keep_prob, 1), dtype=tf.float32)

            dropout_check = self.keep_prob <= tf.constant(1.0)

            dropout = tf.cond(
                dropout_check, dropout_check_true, dropout_check_false)

            self.activations = tf.nn.dropout(self.input_tensor,
                                             keep_prob=dropout,
                                             noise_shape=self.noise_shape)
            tf.summary.histogram('activations', self.activations)

        return self.activations

    def clean(self):
        self.activations = None
        self.R = None

    def _simple_lrp(self, R):
        self.R = R
        return self.R

    def _epsilon_lrp(self, R, epsilon):
        self.R = R
        return self.R

    def _flat_lrp(self, R):
        self.R = R
        return self.R

    def _ww_lrp(self, R):
        self.R = R
        return self.R

    def _alphabeta_lrp(self, R, alpha):
        self.R = R
        return self.R
