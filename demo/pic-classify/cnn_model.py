# coding=utf-8

import tensorflow as tf


class cnn_model(object):
    pic_width = 48
    pic_height = 48
    num_classes = 3
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.pic_width, self.pic_height])
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        self.cnn()

    def cnn(self):
        input = tf.expand_dims(self.x, -1)

        with tf.name_scope("cnn1"):
            w = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[32]), name="B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            act = tf.nn.relu(conv + b)
            pool1 = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope("cnn2"):
            w2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[64]), name="B")
            conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="SAME")
            act2 = tf.nn.relu(conv2 + b2)
            pool2 = tf.nn.max_pool(act2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope("dense"):
            fc = tf.layers.dense(tf.layers.flatten(pool2), 128, name='dense')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

        with tf.name_scope("softmax"):
            self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.y_, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
