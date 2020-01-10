import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)

tf.print(hello)
tf.print([a, b, c])
