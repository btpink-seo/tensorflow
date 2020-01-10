import tensorflow as tf

x_data = tf.cast([[1, 2, 3], [4, 5, 6]], tf.float32)

# 변수
# 그래프를 최적화하는 용도로 텐서플로가 학습한 결과를 갱신하기 위해 사용하는 변수
W = tf.Variable(tf.random.normal([3, 2]))
b = tf.Variable(tf.random.normal([2, 1])) # 정규분포의 무작위 값으로 초기화

print("=== x_data ===")
tf.print(x_data)
print("=== W ===")
tf.print(W)
print("=== b ===")
tf.print(b)
print("=== expr ===")
tf.print(tf.matmul(x_data, W) + b)
