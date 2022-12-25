# 損失函數就是模型預測的值與真實值之間的差距，其梯度就是每個參數求的偏導數。
import tensorflow as tf
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

# 构建梯度环境
with tf.GradientTape() as tape:
  tape.watch([w, a])  # 将w加入梯度跟踪列表
  y = a * w**2 + b * w + c  # 偏導數為2𝑎+𝑏

[dy_dw, dy_da] = tape.gradient(y, [w, a])
print(dy_dw, dy_da)  # 打印出w對y的偏导数、a對y的偏导数（偏微分）

# 偏微分是在一個曲面中，找一條切線並求出它的斜率。
# 斜率作為
