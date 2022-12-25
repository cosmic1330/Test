import tensorflow as tf
import timeit

# 1.创建输入张量
a = tf.constant(2.)
b = tf.constant(4.)
print(a + b)

# 2.在cpu和gpu上建立兩個矩陣
with tf.device('/cpu:0'):
  cpu_a = tf.random.normal([1, 10])
  cpu_b = tf.random.normal([10, 1])

with tf.device('/gpu:0'):
  gpu_a = tf.random.normal([1, 10])
  gpu_b = tf.random.normal([10, 1])

# 3.測量運算時間


def cpu_run():
  with tf.device('/cpu:0'):
    return tf.matmul(cpu_a, cpu_b)


def gpu_run():
  with tf.device('/gpu:0'):
    return tf.matmul(gpu_a, gpu_b)

# 第一次计算需要热身，避免将初始化阶段时间结算在内
cpu_time = timeit.timeit(cpu_run, number=1) 
gpu_time = timeit.timeit(gpu_run, number=1)
print('warmup:', cpu_time, gpu_time)
# 计算10次，取平均时间
cpu_time = timeit.timeit(cpu_run, number=10) 
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)