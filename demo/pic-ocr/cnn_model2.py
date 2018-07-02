# coding=utf-8

import tensorflow as tf


class cnn_model(object):
    pic_width = 32
    pic_height = 128
    nums = 5
    classes = 10
    learning_rate = 0.002  # 学习率

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.pic_width, self.pic_height])
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, self.nums * self.classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.cnn()

    def cnn(self):
        input = tf.expand_dims(self.x, -1)

        with tf.name_scope("cnn1"):
            w = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[32]), name="B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            conv = tf.layers.batch_normalization(conv)
            act = tf.nn.relu(conv + b)
            pool1 = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool1 = tf.nn.dropout(pool1, self.keep_prob)

        with tf.name_scope("cnn2"):
            w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[64]), name="B")
            conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="SAME")
            conv2 = tf.layers.batch_normalization(conv2)
            act2 = tf.nn.relu(conv2 + b2)
            pool2 = tf.nn.max_pool(act2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool2 = tf.nn.dropout(pool2, self.keep_prob)

        with tf.name_scope("cnn3"):
            w3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name="W")
            b3 = tf.Variable(tf.constant(0.1, shape=[64]), name="B")
            conv3 = tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding="SAME")
            conv3 = tf.layers.batch_normalization(conv3)
            act3 = tf.nn.relu(conv3 + b3)
            pool3 = tf.nn.max_pool(act3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool3 = tf.nn.dropout(pool3, self.keep_prob)

        with tf.name_scope("cnn4"):
            w4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1), name="W")
            b4 = tf.Variable(tf.constant(0.1, shape=[64]), name="B")
            conv4 = tf.nn.conv2d(pool3, w4, strides=[1, 1, 1, 1], padding="SAME")
            conv4 = tf.layers.batch_normalization(conv4)
            act4 = tf.nn.relu(conv4 + b4)
            pool4 = tf.nn.max_pool(act4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool4 = tf.nn.dropout(pool4, self.keep_prob)
        with tf.name_scope("dense"):
            fc = tf.layers.dense(tf.layers.flatten(pool4), 512, name='dense')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

        with tf.name_scope("softmax"):
            self.logits = tf.layers.dense(fc, self.nums * self.classes, name='fc2')
            predict = tf.reshape(self.logits, [-1, self.nums, self.classes])
            self.y_pred_cls = tf.argmax(predict, 2)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(tf.reshape(self.y_, [-1, self.nums, self.classes]), 2))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
