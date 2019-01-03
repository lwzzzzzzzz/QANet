import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention

class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo = False, graph = None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit],"context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit],"question")
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit],"context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit],"question_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index2")
            else:
                self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next() # 400 50 30 limited长度
                                                                                    # 并且返回的是batch_size个打包好的examples
                                                                                    # 这里的qa_id是train_eval中的1 2 3 4...这些
            # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                word_mat, dtype=tf.float32), trainable=False) # word_embedding为pre-train不可训练 300d
            self.char_mat = tf.get_variable(
                "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32)) # char_embedding是可训练的 200d

            self.c_mask = tf.cast(self.c, tf.bool) # c中的0被赋值给c_mask为False
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1) # 计算batch内除去pad的每句话的真实长度
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

            if opt: # 最优化这些tensor，slice掉了不需要的pad，减少了计算量
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len) # c_maxlen为该batch内N个context最大长度，之后的slice把不必要计算的pad去掉
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen]) # 从第0行第0列，slice [N, self.c_maxlen]的形状
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen]) # 同样start和end不可能在pad，不如去掉，减少计算量
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1]) # 计算出batch内每个单词的长度，并展开
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            self.forward()
            total_params()

            if trainable:
                # lr随着global_step变化
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(
                    gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads
        # N:batch_siez  PL:c_maxlen  QL:q_maxlen  CL:char_limit-16  d:hidden  dc:char_dim  nh:num_heads
        with tf.variable_scope("Input_Embedding_Layer"):
            # char embedding
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.ch), [N * PL, CL, dc]) # [N * PL, CL, dc]
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            # Bidaf style conv-highway encoder
            # 为什么说word embedding是GloVe预训练好的，不可训练；而char embedding是可训练的，明明都是从xxx_emb.json文件读取？？？
            # 因为：char embedding后面又接了一个conv，来训练它，我们把conv后的认为是char embedding，与word级别concat。
            ch_emb = conv(ch_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = None) # [N * PL, CL-kernel_size+1, d] //VALID卷积
            qh_emb = conv(qh_emb, d,  # reuse为True，表示要重复使用之前的变量，需要使用和前面一样的name
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = True)

            ch_emb = tf.reduce_max(ch_emb, axis = 1) # [N * PL, 1, d]
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]]) # [N ,PL, d]
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout) # [N, PL, glove_dim]
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)  # [N, PL, glove_dim+d] //glove_dim==300 d=96
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None) # [N, PL, d]
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb, # [N, PL, d]
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask, #
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout)
            q = residual_block(q_emb,  # [N, QL, d]
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.q_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.q_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = 1.0 - self.dropout) # [N, PL, QL]
            mask_q = tf.expand_dims(self.q_mask, 1) # [N, 1, self.q_maxlen]
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q)) # mask_logits针对question中的pad进行mask，在S的QL维度上，pad位置都给了很小的数，softmax后为0
            mask_c = tf.expand_dims(self.c_mask, 2) # 同理于上面
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), dim = 1),(0,2,1)) # S_T --> [N, QL, PL]
            self.c2q = tf.matmul(S_, q) # [N, PL, QL] mul [N, QL, d] --> [N, PL, d]
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c) # ([N, PL, QL] mul [N, QL, PL] --> [N, PL, PL]) mul (N, PL, d) --> [N, PL, d]
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c] # concat起来 mat1*mat2 表示对应位置点积

        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis = -1) # [N, PL, 4d]
            self.enc = [conv(inputs, d, name = "input_projection")] # [[N, PL, d]]
            for i in range(3):
                if i % 2 == 0: # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append( # enc.append操作保存QAnet中的3个stacked model encoder blocks的输出，分别为enc[1],[2],[3]，enc[0]为输入
                    residual_block(self.enc[i],
                        num_blocks = 7,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = self.c_mask,
                        num_filters = d,
                        num_heads = nh,
                        seq_len = self.c_len,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None, # 参数也都是复用的
                        dropout = self.dropout)
                    )

        with tf.variable_scope("Output_Layer"):
            # [N, PL, 2d] --> linear proj [N, PL, 1] --> squeeze [N, PL]
            start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "start_pointer"),-1) # no bias
            end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
            self.logits = [mask_logits(start_logits, mask = self.c_mask), # context中被pad的词肯定不是start或end，所以mask掉
                           mask_logits(end_logits, mask = self.c_mask)]

            logits1, logits2 = [l for l in self.logits]

            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2), # [N, PL, 1] mul [N, 1, PL] --> [N, PL, PL]
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, config.ans_limit)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1) # yp2/1 meaning????
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            # tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)相当于做了N个example的交叉熵，--> [N]
            # 再tf.reduce_mean，对N个example求平均，得到loss -->scale
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

        if config.l2_norm is not None:
            # 详细见keith blog https://blog.csdn.net/u012436149/article/details/70264257
            # 本代码在layers.py中regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)定义好了正则化方法，
            # tf.contrib.layers.apply_regularization(regularizer, variables) 来应用在参数上  //variables为需要应用的参数表 //regularizer表示用的正则化方法
            # collection提供了一个全局的存储器，get_collection来获取存储器里的东西，以list返回
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables()) #移动平均应用到所有可训练的变量上
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss) # 并且在更新loss之前，先将移动平均应用到各可训练权重上

                self.assign_vars = []
                for var in tf.global_variables(): # tf中，不论get_variable还是Variable，当collection变量没有赋值时，默认是collections=[tf.GraphKeys.GLOBAL_VARIABLES]
                    v = self.var_ema.average(var)
                    if v: # 当没有应用EMA的variable的v为NONE，有没有应用依据在var_ema.apply(...),在...里面表示应用了
                        # 为什么要加上tf.assign(var,v)，而不直接append(v)原因在于v = self.var_ema.average(var)并不是Graph的操作
                        # 不受tf.control_dependencies([ema_op])限制
                        self.assign_vars.append(tf.assign(var,v)) # 获取当前迭代次数的经过ExponentialMovingAverage后的variable值

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
