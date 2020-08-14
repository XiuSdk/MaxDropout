# !/usr/bin/env python
# coding: utf-8

import tensorflow as tf

class MaxDropout(tf.keras.layers.Layer):
        """MaxDropout: Deep Neural Network Regularization Based on Maximum Output Values
    (https://arxiv.org/abs/2007.13723)
    """

    def __init__(self, rate=0.3, trainable=True, name=None, **kwargs):
        super(MaxDropout, self).__init__(name=name, trainable=trainable, **kwargs)
        if rate < 0 or rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(rate))
        self.rate = 1. - rate

    def call(self, inputs, training=None):
        if training:
            min_in = tf.math.reduce_min(inputs)
            max_in = tf.math.reduce_max(inputs)
            up = inputs - min_in
            divisor = max_in - min_in
            inputs_out = tf.math.divide_no_nan(up, divisor)
            return tf.where(inputs_out > self.rate, tf.zeros_like(inputs), inputs_out)
        else:
            return inputs
 