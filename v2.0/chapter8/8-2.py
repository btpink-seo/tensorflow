import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# parameter
FILE_NAME = Path(__file__).stem
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.2
N_HIDDEN = 256
N_INPUT = 28 * 28
BATCH_SIZE = 100
EPOCHS = 5
SAMPLE_SIZE = 10

# set mnist data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape((-1, 28 * 28)) / 255.0, x_test.reshape((-1, 28 * 28)) / 255.0
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
total_batch = int(len(x_train) / BATCH_SIZE)

# model
with tf.name_scope("encoder"):
    W_encode = tf.Variable(tf.random.normal([N_INPUT, N_HIDDEN], dtype=tf.float64), name="W_encode")
    b_encode = tf.Variable(tf.random.normal([N_HIDDEN], dtype=tf.float64), name="b_encode")

    @tf.function
    def encoder(x):
        return tf.nn.sigmoid(tf.add(tf.matmul(x, W_encode), b_encode))

with tf.name_scope("decoder"):
    W_decode = tf.Variable(tf.random.normal([N_HIDDEN, N_INPUT], dtype=tf.float64), name="W_decode")
    b_decode = tf.Variable(tf.random.normal([N_INPUT], dtype=tf.float64), name="b_decode")

    @tf.function
    def decoder(x):
        return tf.nn.sigmoid(tf.add(tf.matmul(x, W_decode), b_decode))

with tf.name_scope("model"):
    @tf.function
    def model(x, rate=0):
        return decoder(encoder(x))

with tf.name_scope("optimizer"):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

# ckpt
checkpoint_directory = "./{}/".format(FILE_NAME)
if not os.path.exists(checkpoint_directory):
    os.mkdir(checkpoint_directory)
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(W_encode=W_encode, b_encode=b_encode, W_decode=W_decode, b_decode=b_decode)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# tensorboard
train_writer = tf.summary.create_file_writer("logs/{}/train".format(FILE_NAME))
test_writer = tf.summary.create_file_writer("logs/{}/test".format(FILE_NAME))

# training
for epoch in range(EPOCHS):
    total_loss = 0
    total_test_loss = 0

    for images, _ in train_ds:
        loss = lambda: tf.reduce_mean(tf.pow(images - model(images), 2))
        optimizer.minimize(loss, var_list=[W_encode, b_encode, W_decode, b_decode])
        total_loss += loss().numpy()

    for test_images, _ in test_ds:
        loss = lambda: tf.reduce_mean(tf.pow(test_images - model(test_images), 2))
        total_test_loss += loss().numpy()

    # save checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)

    # save tensorboard
    with train_writer.as_default():
        tf.summary.scalar("loss", (total_loss / total_batch), step=checkpoint.save_counter)
    with test_writer.as_default():
        tf.summary.scalar("loss", (total_test_loss / total_batch), step=checkpoint.save_counter)

    template = 'epoch: {}, loss: {}, test loss: {}'
    print(template.format(checkpoint.save_counter.numpy(), total_loss / total_batch, total_test_loss / total_batch))

# plot
samples = model(x_test[:SAMPLE_SIZE])
fig, ax = plt.subplots(2, SAMPLE_SIZE, figsize=(SAMPLE_SIZE, 2))

for i in range(SAMPLE_SIZE):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(x_test[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()
