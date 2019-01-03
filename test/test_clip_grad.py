import tensorflow as tf
import numpy as np

a1 = np.array([[2., 6.],[2., 4.]])
a2 = np.array([[2., 5.],[2., 4.]])
tt = tf.norm(a1,ord = 2)
aa = a1*5/tt
# print(tt)
# print(aa)
b1 = tf.clip_by_norm(a1, 5)
# b2 = tf.clip_by_global_norm(a, 5)
c = tf.norm([3., 4.])
with tf.Session() as sess:
    print(sess.run(b1))
    print(sess.run(aa))
    print(sess.run(c))