import tensorflow as tf

a = tf.constant([[1,2,3],[4,5,6]])

print(type(a.get_shape().as_list()))
print(type(a.shape))
print(type(tf.shape(a)))
print(tf.rank(a))
print(a.get_shape().ndims)