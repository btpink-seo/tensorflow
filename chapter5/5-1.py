import tensorflow as tf
import numpy as np

# unpact=True 행렬을 뒤바꿈
data = np.loadtxt('./5-1.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# 학습횟수 카운트
global_step = tf.Variable(0, trainable=False, name='global_step')

# X: 입력층, Y: 출력층
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# [입력층(특징 수), 출력층(레이블 수)] -> [2, 3]
# 2 -> 특징 수
# 10 -> 은닉층1의 뉴런 수
# 20 -> 은닉층2의 뉴런 수
# 3 -> 분류 수
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

# 교차 엔트로피 함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 경사하강법
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

# 세션 초기화
# model을 불러옴
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

# model폴더에 기존 모델이 있으면 불러오고 아니면 초기화 시킴
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())


# 학습
for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d' % sess.run(global_step), 'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    saver.save(sess, './model/dnn.ckpt', global_step=global_step)

# 결과
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실측값 : ', sess.run(target, feed_dict={Y: y_data}))

# 정확도
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
