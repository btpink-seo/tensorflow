import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# parameter
FILE_NAME = Path(__file__).stem
LEARNING_RATE = 0.0002
DROPOUT_RATE = 0.2
N_HIDDEN = 256
N_INPUT = 28 * 28
N_NOISE = 128
BATCH_SIZE = 100
EPOCHS = 100
SAMPLE_SIZE = 10

# set mnist data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = x_train.reshape((-1, 28 * 28)) / 255.0, tf.one_hot(y_train, 10)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
total_batch = int(len(x_train) / BATCH_SIZE)

with tf.name_scope("Generator"):
    G_W1 = tf.Variable(tf.random.normal([N_NOISE, N_HIDDEN], stddev=0.01, dtype=tf.float64), name="G_W1")
    G_b1 = tf.Variable(tf.zeros([N_HIDDEN], dtype=tf.float64), name="G_b1")
    G_W2 = tf.Variable(tf.random.normal([N_HIDDEN, N_INPUT], stddev=0.01, dtype=tf.float64), name="G_W2")
    G_b2 = tf.Variable(tf.zeros([N_INPUT], dtype=tf.float64), name="G_b2")

    @tf.function
    def generator(noise_z):
        hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
        output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
        return output

    @tf.function
    def model_G(noise):
        return tf.math.log(discriminator(generator(noise)))

with tf.name_scope("Discriminator"):
    D_W1 = tf.Variable(tf.random.normal([N_INPUT, N_HIDDEN], stddev=0.01, dtype=tf.float64), name="D_W1")
    D_b1 = tf.Variable(tf.zeros([N_HIDDEN], dtype=tf.float64), name="D_b1")
    D_W2 = tf.Variable(tf.random.normal([N_HIDDEN, 1], stddev=0.01, dtype=tf.float64), name="D_W2")
    D_b2 = tf.Variable(tf.zeros([1], dtype=tf.float64), name="D_b2")

    @tf.function
    def discriminator(inputs):
        hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
        return output

    @tf.function
    def model_D(images, noise):
        return tf.math.log(discriminator(images)) + tf.math.log(1 - discriminator(generator(noise)))

with tf.name_scope("noise"):
    @tf.function
    def get_noise(batch_size, n_noize):
        return np.random.normal(size=(batch_size, n_noize))

with tf.name_scope("optimizer"):
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# ckpt
with tf.name_scope("ckpt"):
    checkpoint_directory = "./{}/".format(FILE_NAME)
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(G_W1=G_W1, G_b1=G_b1, G_W2=G_W2, G_b2=G_b2, D_W1=D_W1, D_b1=D_b1, D_W2=D_W2, D_b2=D_b2)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# tensorboard
with tf.name_scope("tensorboard"):
    train_writer = tf.summary.create_file_writer("logs/{}/train".format(FILE_NAME))
    test_writer = tf.summary.create_file_writer("logs/{}/test".format(FILE_NAME))

# training
for epoch in range(EPOCHS):
    for images, _ in train_ds:
        noise = get_noise(BATCH_SIZE, N_NOISE)
        loss_D = lambda: -tf.reduce_mean(model_D(images, noise))
        loss_G = lambda: -tf.reduce_mean(model_G(noise))

        optimizer_D.minimize(loss_D, var_list=[D_W1, D_b1, D_W2, D_b2])
        optimizer_G.minimize(loss_G, var_list=[G_W1, G_b1, G_W2, G_b2])

    # save checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)

    # save tensorboard
    with train_writer.as_default():
        tf.summary.scalar("loss_G", loss_G().numpy(), step=checkpoint.save_counter)
        tf.summary.scalar("loss_D", loss_D().numpy(), step=checkpoint.save_counter)

    print('epoch: {}, loss_D: {}, loss_G: {}'.format(checkpoint.save_counter.numpy(), loss_G().numpy(), loss_D().numpy()))

    if epoch == 0 or (epoch + 1) % 10 == 0:
        noise = get_noise(SAMPLE_SIZE, N_NOISE)
        samples = generator(noise)

        fig, ax = plt.subplots(1, SAMPLE_SIZE, figsize=(SAMPLE_SIZE, 1))
        for i in range(SAMPLE_SIZE):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        sample_dir = "./samples1/"
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        plt.savefig('{}{}.png'.format(sample_dir, str(checkpoint.save_counter.numpy()).zfill(3)), bbox_inches='tight')
        plt.close(fig)
