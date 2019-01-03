import tensorflow as tf

batch_size = 5
time_step = 7
depth = 64
inputs = tf.Variable(tf.random_normal([batch_size, time_step, depth]))

# tf.nn.dynamic_rnn例子
# inputs = tf.unstack(inputs, axis=1) #unstack的不能给dynamic_rnn当输入，服了。。。
units = [20,20,20]
fw_cells = [tf.nn.rnn_cell.BasicLSTMCell(unit) for unit in units]  # LSTM层
bw_cells = [tf.nn.rnn_cell.GRUCell(unit) for unit in units] # 后向LSTM层
fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)
output, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.shape(output)))
    print(sess.run(tf.shape(output_state_fw)))
    print(sess.run(tf.shape(output_state_bw)))
