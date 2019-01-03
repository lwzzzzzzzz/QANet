import tensorflow as tf

a = tf.constant([[1,2,3],[0,0,0]], dtype=tf.float32)

b = tf.cast(a, tf.bool)
c = tf.reshape(tf.cast(b, tf.int32), [-1])
# c = tf.slice(a, [0,1], [2,2])
with tf.Session() as sess:
    print(sess.run(c))