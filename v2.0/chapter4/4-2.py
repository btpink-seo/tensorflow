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
W = tf.Variable(tf.random.uniform([2, 3], -1., 1.), trainable=True, dtype=tf.float32)
b = tf.Variable(tf.zeros([3]), trainable=True, dtype=tf.float32)

# 활성화 함수
@tf.function
def hypothesis(x):
    return tf.nn.relu(tf.add(tf.matmul(x, W), b))

model = hypothesis(x_data)

# 교차 엔트로피 함수
loss = lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=hypothesis(x_data)))

# 경사하강법
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 학습
for step in range(1000):
    optimizer.minimize(loss, var_list=[W, b])

    if (step + 1) % 10 == 0:
        print(step + 1, loss().numpy())

# 결과
prediction = tf.argmax(tf.nn.softmax(model), axis=1)
target = tf.argmax(y_data, axis=1)
print('예측값 : ', prediction.numpy())
print('실측값 : ', target.numpy())

# 정확도
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % accuracy.numpy())
