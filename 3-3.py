import tensorflow as tf
# 선형회귀모델

# x_data와 x_data의 상관관계를 파악
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# -1.0 부터 1.0사이의 균등분포를 가진 무작위 값으로 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 자료를 입력받을 플레이스홀더 설정
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

print(X)
print(Y)

# X와 Y의 상관관계를 분석하기 위한 수식
# X가 주어졌을 때 Y를 만들어 낼 수 있는 W와 b를 찾아내겠다
# W : 가중치, b : 편향
hypothesis = W * X + b

# 손실값을 계산하는 함수
# 손실값 : 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타냄
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법 최적화 함수를 이용해 손실값을 최소화하는 연산 그래프를 생성
# learning_rate : 학습을 얼마나 급격히 할 것인가 -> 하이퍼파라메터
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # 학습
  for step in range(100):
    _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
    print(step, cost_val, sess.run(W), sess.run(b))

  # 예측
  print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
  print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
