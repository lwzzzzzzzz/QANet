import tensorflow as tf

a = tf.Variable(1.0)
b = tf.Variable(2.0)
add = tf.add(a,b)
mul = tf.multiply(a, b)
group1 = tf.group(add, mul)
tuple1 = tf.tuple([add, mul])

d = tf.assign(a, 4.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run((d, a)))