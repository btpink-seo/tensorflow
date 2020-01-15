import tensorflow as tf
import numpy as np
import os

# unpact=True 행렬을 뒤바꿈
data = np.loadtxt('./5-1.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# [입력층(특징 수), 출력층(레이블 수)] -> [2, 3]
# 2 -> 특징 수
# 10 -> 은닉층1의 뉴런 수
# 20 -> 은닉층2의 뉴런 수
# 3 -> 분류 수
W1 = tf.Variable(tf.random.uniform([2, 10], -1., 1.), trainable=True, dtype=tf.float32)
b1 = tf.Variable(tf.zeros([10]), trainable=True, dtype=tf.float32)

W2 = tf.Variable(tf.random.uniform([10, 20], -1., 1.), trainable=True, dtype=tf.float32)
b2 = tf.Variable(tf.zeros([20]), trainable=True, dtype=tf.float32)

W3 = tf.Variable(tf.random.uniform([20, 3], -1., 1.), trainable=True, dtype=tf.float32)
b3 = tf.Variable(tf.zeros([3]), trainable=True, dtype=tf.float32)

@tf.function
def hidden1(x):
    return tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

@tf.function
def hidden2(x):
    return tf.nn.relu(tf.add(tf.matmul(x, W2), b2))

@tf.function
def foward(x):
    return tf.add(tf.matmul(x, W3), b3)

# 교차 엔트로피 함수
loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=foward(hidden2(hidden1(x_data)))))

# 경사하강법
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

checkpoint_directory = "./model/"
if not os.path.exists(checkpoint_directory):
    os.mkdir(checkpoint_directory)

checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

checkpoint = tf.train.Checkpoint(W1=W1, W2=W2, W3=W3, b1=b1, b2=b2, b3=b3)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# 학습
for step in range(2):
    optimizer.minimize(loss, var_list=[W1, b1, W2, b2, W3, b3])
    print('Loss: %.3f' % loss().numpy())
    checkpoint.save(file_prefix=checkpoint_prefix)

# 결과
prediction = tf.argmax(foward(hidden2(hidden1(x_data))), axis=1)
target = tf.argmax(y_data, axis=1)
print('예측값 : ', prediction.numpy())
print('실측값 : ', target.numpy())

# 정확도
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % accuracy.numpy())
