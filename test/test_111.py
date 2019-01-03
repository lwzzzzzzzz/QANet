import numpy as np
import tensorflow as tf

x = np.reshape(np.arange(6), [2,3])
a = tf.zeros([2,3])
b = tf.placeholder(dtype=tf.float32, shape=[2,3])
c = tf.add(a, b)
with tf.Session() as sess:
    print(sess.run(b, feed_dict={b:x}))