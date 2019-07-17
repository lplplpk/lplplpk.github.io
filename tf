#with tf.Session() as sess:为上下文管理器
#tf.InteractiveSession()是一种交互式的session方式，它让自己成为了默认的session，也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来，这就是默认的好处
#这样的话就是run()和eval()函数可以不指明session啦。
#而with tf.Session().as_default()也为创建一个默认会话，与with tf.Session() as sess 的区别在于上下文退出管理器时并不会关闭会话，依旧调用eval,run。
import tensorflow as tf
import numpy as np
a=tf.constant([[1., 2., 3.],[4., 5., 6.]])
b=np.float32(np.random.randn(3,2))
c=tf.matmul(a,b)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    print(sess.run(c))
print(sess.run(c))
#会报错，修改方法为用tf.InteractiveSession(),或者使用tf.Session().as_default()
a=tf.constant([[1., 2., 3.],[4., 5., 6.]])
b=np.float32(np.random.randn(3,2))
c=tf.matmul(a,b)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(c)
sess.close()
#不开启会话要手动关闭会话释放资源
##############################

##############################
#!!!tensorflow中所有的变量都需要经过初始化，第一步为定义初始化，第二步为运行初始化
 init=tf.global_variables_initializer()
 with tf.Session() as sess:
    sess.run(init)
##############################   
 #############################  
 ##为了使变量b能接受任意值，引入tf.placehoder()，与此同时sess.run时需要用上feed
 b=tf.placeholder(tf.float32,[None,1])
 #.....
 sess.run(a,feed_dict={b:np.arrange(0,10)})
 ######################################
 
 #####################################
#在定义完变量，超参数，和网络结构后，定义损失，并且定义优化器
cross_entropy=tf.nn.softmax_cross_entropy_with_logits() #不同场合下使用不同的函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimizer(cross_entropy)

# tf.get_variable(name,shape,initializer),最后一个参数可以指定不同的变量以不同的方式初始化
#1.tf.constant_initializer:常量的初始化函数
#2.tf.random_normalize_initializer:正态分布的初始化
#3.tf.truncated_normalize_initializer:截取的正态分布
#4 tf.random_uniform_initializer:均匀分布
#5 tf.zeros_initializer:全零分布，tf.ones_initializer:全一分布
#6 tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
#7 tf.contrib.layers.xavier_initializer：对权重的初始化，该初始化器旨在使所有层中的梯度比例保持大致相同





 
 
