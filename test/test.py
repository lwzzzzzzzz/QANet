import tensorflow as tf

a = tf.get_variable('a',initializer=tf.ones([2,3]))
c = tf.Variable(a)
b = tf.assign(a, a)

print(a)