import tensorflow as tf

sess = tf.Session()
a = tf.get_variable("a", [3, 3, 32, 64], initializer=tf.random_normal_initializer())
b = tf.get_variable("b", [64], initializer=tf.random_normal_initializer())
# collections=None等价于 collection=[tf.GraphKeys.GLOBAL_VARIABLES]
c = tf.add_to_collection("test", [2.1]) # 函数自动创建collection
gv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # tf.get_collection(collection_name)返回某个collection的列表
d = tf.get_collection('test')
print(d)
print(gv)
for var in gv:
    print(var is a)
    print(var.get_shape())
