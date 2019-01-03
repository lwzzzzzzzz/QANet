import tensorflow as tf
import numpy as np

x = 'i looove you'
x_ids = np.array([1, 2, 3])
label = 1

writer = tf.python_io.TFRecordWriter('./test.tfrecords')
# 这两个features |features = tf.train.Features| 都要加s！！！否则报错
for i in range(3):
    one_record = tf.train.Example(features = tf.train.Features(feature = {
                                "x_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(x+' this is'+str(i), 'utf-8')])),
                                "x_ids":tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.append(x_ids, np.array([i])).tostring()])),
                                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label+i]))
        }))

    writer.write(one_record.SerializeToString())
writer.close()
