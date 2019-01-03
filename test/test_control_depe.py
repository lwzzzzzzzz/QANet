import tensorflow as tf

w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)
# tf.train.Optimizer.minimize()
ema_op = ema.apply([update])
with tf.control_dependencies([ema_op]):
    # ema_val = ema.average(update)
    ema_val = tf.identity(ema.average(update))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        print(sess.run([ema_val]))
# g = tf.Graph()
# with g.as_default():
#     x = tf.Variable(1.0, name='x')
#     x_plus_1 = tf.assign_add(x, 1, name='x_plus')
#
#     with tf.control_dependencies([x_plus_1]):
#         y = x
#         z = tf.identity(x, name='z_added')
#
#     init = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         sess.run(init)
#         for i in range(5):
#             print(sess.run(y))
#             print(sess.run(z)) # 输出 2,3,4,5,6
#
#
#
#         # 如果改为输出 print(sess.run(y)) ,则结果为 1,1,1,1,1
