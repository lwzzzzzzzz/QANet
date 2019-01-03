import tensorflow as tf

a = tf.Variable(1.0)
b = tf.assign_add(a, 10.0, name='up_a')

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './test2.ckpt')
    for i in range(10):
        tt = sess.run(b)
        print(tt)
        # if i == 4:
        #     saver.save(sess, './test1.ckpt')
        # if i == 9:
        #     saver.save(sess, './test2.ckpt')
