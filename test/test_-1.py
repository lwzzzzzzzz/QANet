import tensorflow as tf

batch_size = 5
max_time = 7
depth = 64
inputs = tf.Variable(tf.random_normal([batch_size, max_time, depth]))


# tf.nn.dynamic_rnn例子
# inputs = tf.transpose(inputs, (1,0,2))
inputs = tf.unstack(inputs, axis=1) #unstack的不能给dynamic_rnn当输入，服了。。。
fw_units = [20,20,20]
bw_units = [20,20,20]

fw_cells = [tf.nn.rnn_cell.LSTMCell(unit) for unit in fw_units]  # 前向LSTM层
bw_cells = [tf.nn.rnn_cell.GRUCell(unit) for unit in bw_units]  # 后向LSTM层
outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, inputs, dtype=tf.float32)
res = tf.stack(outputs, 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.shape(outputs)))
    print(sess.run(tf.shape(res)))

    print(sess.run(tf.shape(output_state_fw)))
    print(sess.run(tf.shape(output_state_bw)))
