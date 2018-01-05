# coding=utf-8
import tensorflow as tf

x = tf.placeholder(dtype=tf.int32)
y_ = tf.placeholder(dtype=tf.int32)

### 优化1这个数字会被阴式转换为tensor，多次调用该加法时会生成多个该tensor，因此提前手动将其转换为tensor可以避免内存溢出
# i = tf.convert_to_tensor(1)
# a = x + i
a = x + 1
t = tf.Print(a, [a, x], message="输出中间数据的内容a,x=")
y = t + 9

sess = tf.InteractiveSession()
print sess.run(y, feed_dict={x: 11})
sess.close()
