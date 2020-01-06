import tensorflow as tf
# 플레이스홀더
# 그래프에 사용할 입력값을 나중에 받기 위해 사용하는 매개변수
X = tf.placeholder(tf.float32, [None, 3])
x_data = [[1, 2, 3], [4, 5, 6]]

# 변수
# 그래프를 최적화하는 용도로 텐서플로가 학습한 결과를 갱신하기 위해 사용하는 변수
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1])) # 정규분포의 무작위 값으로 초기화
expr = tf.matmul(X, W) + b # 행렬곱

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수들을 초기화(W, b)

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
# feed_dict : 그래프를 실행할 때 사용할 입력값을 지정
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()