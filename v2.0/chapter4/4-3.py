import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]], dtype='float32')

# 원-핫 인코딩
y_data = np.array([
    [1, 0, 0], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
], dtype='float32')

# [입력층(특징 수), 출력층(레이블 수)] -> [2, 3]
W1 = tf.Variable(tf.random.uniform([2, 10], -1., 1.), trainable=True, dtype=tf.float32)
b1 = tf.Variable(tf.zeros([10]), trainable=True, dtype=tf.float32)
W2 = tf.Variable(tf.random.uniform([10, 3], -1., 1.), trainable=True, dtype=tf.float32)
b2 = tf.Variable(tf.zeros([3]), trainable=True, dtype=tf.float32)

@tf.function
def hidden(x):
    return tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

@tf.function
def foward(x):
    return tf.add(tf.matmul(x, W2), b2)

# 교차 엔트로피 함수
loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=foward(hidden(x_data))))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 학습
for step in range(100):
    optimizer.minimize(loss, var_list=[W1, b1, W2, b2])

    if (step + 1) % 10 == 0:
        print(step + 1, loss().numpy())

# 결과
prediction = tf.argmax(foward(hidden(x_data)), axis=1)
target = tf.argmax(y_data, axis=1)
print('예측값 : ', prediction.numpy())
print('실측값 : ', target.numpy())

# 정확도
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % accuracy.numpy())
