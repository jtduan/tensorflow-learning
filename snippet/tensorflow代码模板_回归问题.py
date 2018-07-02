# coding=utf-8
import tensorflow as tf
import numpy as np

batch_size = 128


def next_batch(batch_size):
    x = []
    y = []
    for _ in range(batch_size):
        x1 = np.random.rand() * 5
        x2 = np.random.rand() * 5
        x3 = np.random.rand() * 10
        x.append([x1, x2, x3])
        y.append([(x1 * x1 + x2 * x2 + x3)])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


##############深度学习模型,本例为1层全连接模型

class DEMO_MODEL(object):

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 3])
        self.y_ = tf.placeholder(tf.float32, [None, 1])

        self.y_pred = self.inference(self.x)  # 模型的one-hot输出
        self.losses = self.loss(self.y_pred, self.y_)
        self.train_op = self.training(self.losses, 0.0001)  # 模型训练步骤

    def inference(self, inputs):
        layer1 = tf.layers.dense(inputs, 200)
        layer1 = tf.nn.sigmoid(layer1)
        # w = tf.Variable(tf.random_normal([3, 1]))
        # b = tf.Variable(tf.random_normal([1, 1]))
        # layer2 = tf.add(tf.matmul(inputs, w), b)
        layer2 = tf.layers.dense(layer1, 1)
        return layer2

    #######定义损失函数
    def loss(self, logits, labels):
        return tf.reduce_sum(tf.pow(logits - labels, 2), name='xentropy_mean')

    ######训练函数
    def training(self, loss, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


def start():
    model = DEMO_MODEL()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for step in xrange(50000):
        input_x, input_y = next_batch(batch_size)
        loss, _, y_pred = sess.run([model.losses, model.train_op, model.y_pred],
                                   feed_dict={model.x: input_x, model.y_: input_y})
        if (step % 100 == 1):
            print "losses=", loss, sess.run(model.y_pred, feed_dict={model.x: [[4, 2, 6]]}), "=", 26

    sess.close()


start()
