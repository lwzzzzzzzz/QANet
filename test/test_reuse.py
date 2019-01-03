import tensorflow as tf

with tf.variable_scope('V1',initializer=tf.zeros([1])):
    a1 = tf.get_variable(name='a1')

with tf.variable_scope('V1', reuse=True):
    a2 = tf.get_variable('a1')
    a3 = tf.Variable(tf.zeros([1]), name='a1')
    a4 = tf.Variable(tf.zeros([1]), name='a1')

print('a1:', a1.name)
print('a2:', a2.name)
print('a3:', a3.name)
print('a4:', a4.name)

