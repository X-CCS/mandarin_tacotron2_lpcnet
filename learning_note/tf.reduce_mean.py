'''
https://blog.csdn.net/dcrmg/article/details/79797826
'''
import tensorflow as tf

x = [[1,2,3],
     [1,2,3]]

xx = tf.cast(x,tf.float32)

mean_all = tf.reduce_mean(xx,keep_dims=False)
mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False) # y轴
mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False) # x轴

with tf.Session() as sess:
     m_a,m_0,m_1 = sess.run([mean_all,mean_0,mean_1])