import tensorflow as tf
x = tf.ones([3,4])
ooo = x.get_shape()[-1]
with tf.variable_scope('test', values=[x]):
    b = tf.get_variable('b', [ooo])
    print(b)
    a = x + 1
    print(x)
print(a)
