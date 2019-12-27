"""
https://blog.csdn.net/UESTC_C2_403/article/details/72190282
"""
import tensorflow as tf
import numpy as np

A = tf.convert_to_tensor(np.array([[1, 1, 2, 4], [3, 4, 8, 5]]))

with tf.Session() as sess:
	print(A.dtype)
	b = tf.cast(A, tf.float32)
	print(b.dtype)