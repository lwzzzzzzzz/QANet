import tensorflow as tf
import numpy as np

# #创建一个Dataset对象
# # tf.data.Dataset.from_tensor_slices(tensor)是将tensor的切片处理成dataset的数据
# # dataset = tf.data.Dataset.from_tensor_slices(np.zeros([4, 10]))
# dataset = tf.data.Dataset.from_tensors(np.zeros([4, 10])) # tf.data.Dataset.from_tensors是将整个tensor当作一个处理
# #创建一个迭代器
# iterator = dataset.make_one_shot_iterator()
# #get_next()函数可以帮助我们从迭代器中获取元素
# element = iterator.get_next()
# #遍历迭代器，获取所有元素
# with tf.Session() as sess:
#     # for i in range(3):
#    print(sess.run(element))
#定义一个生成器
# def data_generator():
#     dataset = np.array(range(8))
#     for i in dataset:
#         yield i, i*i
# #接收生成器，并生产dataset数据结构
# dataset = tf.data.Dataset.from_generator(data_generator, (tf.int32, tf.float32)) # 多增加一个tf.dtype的参数指示返回的数据类型
# def parse(x):
#     return x-10
# dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
# dataset1 = tf.data.Dataset.from_tensor_slices([6,7,8])
# # dataset = dataset.concatenate(dataset1).filter(lambda x:x>3)
# dataset = dataset.concatenate(dataset1)
# dataset = dataset.map(parse)
# dataset=dataset.batch(3, drop_remainder=True).repeat().shuffle(1000)
# # dataset.padded_batch(batch_size,padded_shapes,..)函数表示原来的每一条data，pad成padded_shapes形状，再把batch_size个组合起来
# # dataset=dataset.padded_batch(2,padded_shapes=[5],padding_values=1)
# # dataset=dataset.padded_batch(1,padded_shapes=[7,7],padding_values=77)
#
#
# iterator = dataset.make_one_shot_iterator()
# element = iterator.get_next()
# with tf.Session() as sess:
#     for i in range(10):
#        print(sess.run(element))

training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).batch(5).repeat()
validation_dataset = tf.data.Dataset.range(50).batch(5).repeat()

# 创建了一个可以重复初始化的iterator，分别在training_init_op和validation_init_op中被初始化
# 且training_dataset和validation_dataset的output_types和output_shapes一样
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
with tf.Session() as sess:
    print(training_dataset.output_shapes)
    for _ in range(20):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(100):
            print(sess.run(next_element))
        print('-------------')
        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
            print(sess.run(next_element))
