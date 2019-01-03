import tensorflow as tf

logits = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
y = tf.nn.softmax(logits)
y_ = tf.constant([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])  # 这个是稀疏的标签
tf_log = tf.log(y)
pixel_wise_mult = tf.multiply(y_, tf_log)

# step4:do cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(pixel_wise_mult, axis=1))

#代码段2，使用tf.nn.softmax_cross_entropy_with_logits算出代价函数
cross_entropy2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))#dont forget tf.reduce_sum()!!

#代码段3，使用tf.nn.sparse_softmax_cross_entropy_with_logits()算出代价函数
# 将标签稠密化
dense_y = tf.arg_max(y_, 1)
cross_entropy3 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dense_y, logits=logits))
##### 唯一的区别是sparse的labels是int类型，而非sparse的labels是one-hot类型。
with tf.Session() as sess:
    result1,result2,result3 = sess.run(
        (cross_entropy,cross_entropy2,cross_entropy3))
    print("method1 : %s" % result1)
    print("method2 : %s" % result2)
    print("method3 : %s" % result3)
