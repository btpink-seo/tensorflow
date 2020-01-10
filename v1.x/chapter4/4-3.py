import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# 원-핫 인코딩
y_data = np.array([
    [1, 0, 0], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# X: 입력층, Y: 출력층
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# [입력층(특징 수), 출력층(레이블 수)] -> [2, 3]
# 2 -> 특징 수
# 10 -> 은닉층의 뉴런 수
# 3 -> 분류 수
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# 활성화 함수
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)

# 교차 엔트로피 함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 경사하강법
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 결과
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실측값 : ', sess.run(target, feed_dict={Y: y_data}))

# 정확도
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
