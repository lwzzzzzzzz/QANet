import tensorflow as tf


a = tf.get_variable('in',[3,4,5], trainable=False) # in [N, T, C]
params = {"inputs": a, "filters": 7, "kernel_size": 1, # 这里的kernel_size是[1, T],1只是说每1个单词进行卷积
          "activation": tf.nn.relu, "use_bias": True}
b = tf.layers.conv1d(**params)
for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
print(b)