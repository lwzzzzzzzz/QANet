import tensorflow as tf
import numpy as np

x = np.arange(4, dtype=np.float32).reshape(2,2) # 使用np来创造两个样本
y_ = np.array([0,1], dtype=np.float32).reshape(2,1) # 使用np来创造两个label
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1),name='w1')
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1),name='w2')
b1 = tf.Variable(tf.zeros([3]),name='b1')
b2 = tf.Variable(tf.zeros([1]),name='b2')

a = tf.nn.relu(tf.matmul(x,w1) + b1)
y = tf.matmul(a, w2) + b2

cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_, name=None)

opt = tf.train.GradientDescentOptimizer(0.1)
grads = opt.compute_gradients(cost)
# for i, (g, v) in enumerate(grads):
#     if g is not None:
#         print(g)
#         grads[i] = (tf.clip_by_norm(g, 5), v)
# print(grads)
gradients, variables = zip(*grads)
print(gradients)
capped_grads, _ = tf.clip_by_global_norm(gradients, 5)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(capped_grads))