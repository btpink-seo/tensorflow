import tensorflow as tf

# 선형회귀모델

# x_data와 y_data의 상관관계를 파악
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# -1.0 부터 1.0사이의 균등분포를 가진 무작위 값으로 초기화
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random.uniform([1], -1.0, 1.0))

# X와 Y의 상관관계를 분석하기 위한 수식
# X가 주어졌을 때 Y를 만들어 낼 수 있는 W와 b를 찾아내겠다
# W : 가중치, b : 편향
@tf.function
def hypothesis(x):
    return W * x + b

# 손실값을 계산하는 함수
# 손실값 : 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타냄
loss = lambda: tf.reduce_mean(tf.square(hypothesis(x_data) - y_data))

# 경사하강법 최적화 함수를 이용해 손실값을 최소화하는 연산 그래프를 생성
# learning_rate : 학습을 얼마나 급격히 할 것인가 -> 하이퍼파라메터
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 학습
for step in range(100):
    optimizer.minimize(loss, var_list=[W, b])
    print(step, loss().numpy(), W.numpy(), b.numpy())

# 예측
print("X: 5, Y:", hypothesis(5).numpy()[0])
print("X: 2.5, Y:", hypothesis(2.5).numpy()[0])
