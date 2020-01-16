import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

file_name = Path(__file__).stem

# set mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape((-1, 28, 28, 1)) / 255.0, x_test.reshape((-1, 28, 28, 1)) / 255.0
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

batch_size = 100
total_batch = int(len(x_train) / batch_size)
EPOCHS = 7
RATE = 0.2

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# model
with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01, dtype=tf.float64), name="W1")
    @tf.function
    def layer1(x):
        L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        return tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01, dtype=tf.float64), name="W2")
    @tf.function
    def layer2(x):
        L2 = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        return tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("fully connected layer"):
    W3 = tf.Variable(tf.random.normal([7 * 7 * 64, 256], stddev=0.01, dtype=tf.float64), name="W3")
    @tf.function
    def connect(x, rate):
        L3 = tf.reshape(x, [-1, 7 * 7 * 64])
        L3 = tf.matmul(L3, W3)
        L3 = tf.nn.relu(L3)
        return tf.nn.dropout(L3, rate)

with tf.name_scope("model"):
    W4 = tf.Variable(tf.random.normal([256, 10], stddev=0.01, dtype=tf.float64), name="model")
    @tf.function
    def model(x, rate=0):
        L3 = connect(layer2(layer1(x)), rate)
        return tf.matmul(L3, W4)

with tf.name_scope("optimizer"):
    optimizer = tf.keras.optimizers.RMSprop()

# save & restore checkpoint
checkpoint_directory = "./{}/".format(file_name)
if not os.path.exists(checkpoint_directory):
    os.mkdir(checkpoint_directory)

checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(W1=W1, W2=W2, W3=W3, W4=W4)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# tensorboard
train_writer = tf.summary.create_file_writer("logs/{}/train".format(file_name))
test_writer = tf.summary.create_file_writer("logs/{}/test".format(file_name))

# 학습
for epoch in range(EPOCHS):
    total_loss = 0
    total_test_loss = 0

    for images, labels in train_ds:
        loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model(images, RATE)))
        optimizer.minimize(loss, var_list=[W1, W2, W3, W4])
        total_loss += loss().numpy()
        is_correct = tf.equal(tf.argmax(model(images), 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float64))

    for test_images, test_labels in test_ds:
        loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_labels, logits=model(test_images)))
        total_test_loss += loss().numpy()
        is_test_correct = tf.equal(tf.argmax(model(test_images), 1), tf.argmax(test_labels, 1))
        test_accuracy = tf.reduce_mean(tf.cast(is_test_correct, tf.float64))

    # save checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)

    # save ckpt
    with train_writer.as_default():
        tf.summary.scalar("loss", (total_loss / total_batch), step=checkpoint.save_counter)
        tf.summary.scalar("accuracy", accuracy*100, step=checkpoint.save_counter)
    with test_writer.as_default():
        tf.summary.scalar("loss", (total_test_loss / total_batch), step=checkpoint.save_counter)
        tf.summary.scalar("accuracy", test_accuracy*100, step=checkpoint.save_counter)

    template = 'epoch: {}, loss: {}, accuracy: {}%, test loss: {}, test accuracy: {}%'
    print(template.format(checkpoint.save_counter.numpy(), total_loss / total_batch, accuracy*100, total_test_loss / total_batch, test_accuracy*100))

# plot
# plot_labels = model(x_test)

# fig = plt.figure()
# for i in range(10):
#     subplot = fig.add_subplot(2, 5, i + 1)
#     subplot.set_xticks([])
#     subplot.set_yticks([])
#     subplot.set_title('%d' % np.argmax(plot_labels[i]))
#     subplot.imshow(x_test[i].reshape((28, 28)))

# plt.show()
