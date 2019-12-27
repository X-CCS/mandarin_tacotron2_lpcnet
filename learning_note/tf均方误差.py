"""
tensorflow--均方误差(MSE, mean squared error)表示方法
https://blog.csdn.net/liuyingying0418/article/details/80070538
"""
import tensorflow as tf

a = tf.constant([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
b = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
# print(a)
# print(b)

# 而如果要得到具体的类似array形式的值，则需要用到sess.run:
# with tf.Session() as sess:
# 	print(sess.run(a))
# 	print(sess.run(b))

mse2 = tf.losses.mean_squared_error(a, b)
with tf.Session() as sess:
	print(sess.run(mse2))
