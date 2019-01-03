import tensorflow as tf

A = tf.constant([[3.,4.],[3.,4.]])
A = tf.unstack(A,axis=0)
print(A)

with tf.Session() as sess:
    t_clip, global_norm = sess.run(tf.clip_by_global_norm(A, 5))
    print(t_clip)
    print(global_norm)