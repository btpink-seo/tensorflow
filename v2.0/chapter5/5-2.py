# tensorboard --logdir=./logs
# http://localhost:6006/

import tensorflow as tf
import numpy as np
import os

# unpact=True 행렬을 뒤바꿈
data = np.loadtxt('./5-1.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random.uniform([2, 10], -1., 1.), trainable=True, dtype=tf.float32, name="W1")
    b1 = tf.Variable(tf.zeros([10]), trainable=True, dtype=tf.float32, name="b1")
    @tf.function
    def hidden1(x):
        return tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random.uniform([10, 20], -1., 1.), trainable=True, dtype=tf.float32, name="W2")
    b2 = tf.Variable(tf.zeros([20]), trainable=True, dtype=tf.float32, name="b2")
    @tf.function
    def hidden2(x):
        return tf.nn.relu(tf.add(tf.matmul(x, W2), b2))

with tf.name_scope("output"):
    W3 = tf.Variable(tf.random.uniform([20, 3], -1., 1.), trainable=True, dtype=tf.float32, name="W3")
    b3 = tf.Variable(tf.zeros([3]), trainable=True, dtype=tf.float32, name="b3")
    @tf.function
    def foward(x):
        return tf.add(tf.matmul(x, W3), b3)

with tf.name_scope("optimizer"):
    loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=foward(hidden2(hidden1(x_data)))))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# save & restore checkpoint
checkpoint_directory = "./model2/"
if not os.path.exists(checkpoint_directory):
    os.mkdir(checkpoint_directory)

checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(W1=W1, W2=W2, W3=W3, b1=b1, b2=b2, b3=b3)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# tensorboard
writer = tf.summary.create_file_writer("./logs")

# 학습
with writer.as_default():
    for step in range(10):
        optimizer.minimize(loss, var_list=[W1, b1, W2, b2, W3, b3])
        print('Loss: %.3f' % loss().numpy())
        checkpoint.save(file_prefix=checkpoint_prefix) # save checkpoint
        # save summary for tensorboard
        tf.summary.scalar("loss", loss(), step=checkpoint.save_counter)
        tf.summary.histogram("W1", W1, step=checkpoint.save_counter)
        writer.flush()

# 결과
prediction = tf.argmax(foward(hidden2(hidden1(x_data))), axis=1)
target = tf.argmax(y_data, axis=1)
print('예측값 : ', prediction.numpy())
print('실측값 : ', target.numpy())

# 정확도
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % accuracy.numpy())
