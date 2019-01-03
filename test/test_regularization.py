import tensorflow as tf
sess=tf.Session()
weight_decay=0.1                                                #(1)定义weight_decay
l2_reg=tf.contrib.layers.l2_regularizer(weight_decay)           #(2)定义l2_regularizer()
tmp=tf.constant([0,1,2,3],dtype=tf.float32)
a=tf.Variable(tmp, name="I_am_a",)  #(3)创建variable，l2_regularizer复制给regularizer参数。
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, a)
          #get_variable和variable_scope两个函数的regularizer参数如果给定了，自动将该项的loss加入到REGULARIZATION_LOSSES
b=tf.get_variable("I_am_b",regularizer=l2_reg,initializer=tmp+1)
# tf.GraphKeys实际就是存储了一堆自己内部维护的collection名字
print("Global Set:")
keys = tf.get_collection("variables")
for key in keys:
  print(key.name)
print("Regular Set:")
keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
for key in keys:
  print(key.name)
print("--------------------")
sess.run(tf.global_variables_initializer())
print(sess.run((a, b)))
reg_set=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)   #(4)则REGULARIAZTION_LOSSES集合会包含所有被weight_decay后的参数和，将其相加
print(reg_set)
l22_loss = tf.contrib.layers.apply_regularization(l2_reg, reg_set)
# l2_loss=tf.add_n(reg_set) # 将一个list所有进行加和
loss2, reg = sess.run((l22_loss,reg_set))
# print("loss=%f" %loss)
print("loss2=%f" %loss2)
print(reg)
"""
此处输出0.7,即:
   weight_decay*sigmal(w*2)/2=0.1*(0*0+1*1+2*2+3*3)/2=0.7
其实代码自己写也很方便，用API看着比较正规。
在网络模型中，直接将l2_loss加入loss就好了。(loss变大，执行train自然会decay)
"""
