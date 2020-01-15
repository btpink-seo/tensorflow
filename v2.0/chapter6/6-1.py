import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# set mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28 * 28) / 255.0, x_test.reshape(-1, 28 * 28) / 255.0
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

batch_size = 100
total_batch = int(len(x_train) / batch_size)
EPOCHS = 15

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# model
with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random.normal([784, 256], stddev=0.01, dtype=tf.float64), name="W1")
    b1 = tf.Variable(tf.zeros([256], dtype=tf.float64), name="b1")
    @tf.function
    def hidden1(x):
        return tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random.normal([256, 256], stddev=0.01, dtype=tf.float64), name="W2")
    b2 = tf.Variable(tf.zeros([256], dtype=tf.float64), name="b2")
    @tf.function
    def hidden2(x):
        return tf.nn.relu(tf.add(tf.matmul(x, W2), b2))

with tf.name_scope("output"):
    W3 = tf.Variable(tf.random.normal([256, 10], stddev=0.01, dtype=tf.float64), name="W3")
    b3 = tf.Variable(tf.zeros([10], dtype=tf.float64), name="b3")
    @tf.function
    def foward(x):
        return tf.add(tf.matmul(x, W3), b3)

with tf.name_scope("optimizer"):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# save & restore checkpoint
checkpoint_directory = "./{}/".format(Path(__file__).stem)
if not os.path.exists(checkpoint_directory):
    os.mkdir(checkpoint_directory)

checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(W1=W1, W2=W2, W3=W3, b1=b1, b2=b2, b3=b3)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# tensorboard
writer = tf.summary.create_file_writer("./logs")

# 학습
with writer.as_default():
    for epoch in range(EPOCHS):
        total_loss = 0
        total_test_loss = 0

        for images, labels in train_ds:
            loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=foward(hidden2(hidden1(images)))))
            optimizer.minimize(loss, var_list=[W1, b1, W2, b2, W3, b3])
            total_loss += loss().numpy()
            is_correct = tf.equal(tf.argmax(foward(hidden2(hidden1(images))), 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float64))

        for test_images, test_labels in test_ds:
            loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=test_labels, logits=foward(hidden2(hidden1(test_images)))))
            total_test_loss += loss().numpy()
            is_test_correct = tf.equal(tf.argmax(foward(hidden2(hidden1(test_images))), 1), tf.argmax(test_labels, 1))
            test_accuracy = tf.reduce_mean(tf.cast(is_test_correct, tf.float64))

        # save checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix)

        # save ckpt
        tf.summary.scalar("loss", (total_loss / total_batch), step=checkpoint.save_counter)
        tf.summary.scalar("test_loss", (total_test_loss / total_batch), step=checkpoint.save_counter)
        tf.summary.scalar("accuracy", accuracy*100, step=checkpoint.save_counter)
        tf.summary.scalar("test_accuracy", test_accuracy*100, step=checkpoint.save_counter)
        tf.summary.histogram("W1", W1, step=checkpoint.save_counter)
        tf.summary.histogram("W2", W1, step=checkpoint.save_counter)
        tf.summary.histogram("W3", W1, step=checkpoint.save_counter)
        tf.summary.histogram("b1", W1, step=checkpoint.save_counter)
        tf.summary.histogram("b2", W1, step=checkpoint.save_counter)
        tf.summary.histogram("b3", W1, step=checkpoint.save_counter)
        writer.flush()

        template = 'epoch: {}, loss: {}, accuracy: {}%, test loss: {}, test accuracy: {}%'
        print(template.format(checkpoint.save_counter.numpy(), (total_loss / total_batch), accuracy*100, total_test_loss / total_batch, test_accuracy*100))
