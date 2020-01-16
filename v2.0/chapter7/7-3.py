import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks, datasets, layers, models, Sequential

file_name = Path(__file__).stem

# set mnist data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images.reshape((-1, 28, 28, 1)) / 255.0, test_images.reshape((-1, 28, 28, 1)) / 255.0

EPOCHS = 3

# 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ckpt
checkpoint_path = "./{}/".format(file_name)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
cp_callback = callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
if os.path.exists("{}/checkpoint".format(checkpoint_path)):
    model.load_weights(checkpoint_path)

# tensorboard
log_dir = "./logs/{}/".format(file_name)
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)

# 학습
model.fit(train_images,
          train_labels,
          epochs=EPOCHS,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback, cp_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# plot
plot_labels = model(test_images)

fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(2, 5, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(plot_labels[i + 33]))
    subplot.imshow(test_images[i + 33].reshape((28, 28)))

plt.show()
