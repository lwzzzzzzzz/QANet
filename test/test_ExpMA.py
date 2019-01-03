import tensorflow as tf
w = tf.Variable(1.0, name='w')
step = tf.Variable(0.0, collections=[tf.GraphKeys.GLOBAL_VARIABLES], name='step')
ema = tf.train.ExponentialMovingAverage(0.9, step)
updata_step = tf.assign_add(step, 10.0, name='up_step')
update = tf.assign_add(w, 1.0, name='up_w')
assign_vars = []
ema_op = ema.apply([w])  # 这句和下面那句不能调换顺序

with tf.control_dependencies([update, updata_step]):
    ema_val = tf.identity(ema.average(w))  # 参数不能是list，有点蛋疼

    #返回一个op,这个op用来更新moving_average,i.e. shadow value
    for var in tf.global_variables():  # tf中，不论get_variable还是Variable，当collection变量没有赋值时，默认是collections=[tf.GraphKeys.GLOBAL_VARIABLES]
        v = ema.average(var)

        if v:
            # 为什么要加上tf.assign(var,v)，而不直接append(v)原因在于v = self.var_ema.average(var)并不是Graph的操作
            # 不受tf.control_dependencies([ema_op])限制
            assign_vars.append(v)  # 获得了当前迭代次数的经过ExponentialMovingAverage后的variable值

# 以 w 当作 key， 获取 shadow value 的值

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        sess.run(ema_op)
        print(sess.run((ema_val, assign_vars, step)))

# 创建一个时间序列 1 2 3 4
#输出：
#1.1      =0.9*1 + 0.1*2
#1.29     =0.9*1.1+0.1*3
#1.561    =0.9*1.29+0.1*4