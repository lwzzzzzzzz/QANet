import tensorflow as tf
import numpy as np
x = [[[1, 2, 3],
     [1, 2, 3]],
     [[3, 4, 5],
      [6, 7, 8]],
     [[3, 4, 5],
      [6, 7, 8]]]

xx = tf.cast(x, tf.float32)
a = tf.nn.moments(xx,axes=[-1],keep_dims=True)
print(a)
# tf.layers.batch_normalization()
# mean_all = tf.reduce_mean(xx, keep_dims=False)
# mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False)
# mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False)
# print(xx)
# print(mean_all)
# print(mean_0)
# print(mean_1)
# b = [1,2,3,4]
# a = list(range(len(b)))
# del a[1]
# print(a)