import tensorflow as tf

a = tf.zeros((10, 10))
a = tf.reshape(a, (-1))
print(a, a.shape)