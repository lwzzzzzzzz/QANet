import tensorflow as tf

def parser(example):
    features = tf.parse_single_example(example,features={
                                                "x_raw": tf.FixedLenFeature([], tf.string),
                                                "x_ids": tf.FixedLenFeature([], tf.string),
                                                "label": tf.FixedLenFeature([], tf.int64)
                                            })
    x_ids = tf.decode_raw(features["x_ids"], tf.int32)
    x_raw = features["x_raw"]
    label = features["label"]
    return x_ids, x_raw, label

dataset1 = tf.data.TFRecordDataset('./test.tfrecords').map(parser).shuffle(10).repeat().batch(2)
dataset2 = tf.data.TFRecordDataset('./test.tfrecords').map(parser).shuffle(10).repeat().batch(3)
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, dataset1.output_types, dataset2.output_shapes)
element = iterator.get_next()

iterator1 = dataset1.make_one_shot_iterator()
iterator2 = dataset2.make_one_shot_iterator()

with tf.Session() as sess:

    for i in range(3):
        dataset1_handle = sess.run(iterator1.string_handle())
        a1, b1, c1 = sess.run(element, feed_dict={handle: dataset1_handle})
        print(a1, b1, c1)
        dataset2_handle = sess.run(iterator2.string_handle())
        a2, b2, c2 = sess.run(element, feed_dict={handle: dataset2_handle})
        print(a2, b2, c2)

