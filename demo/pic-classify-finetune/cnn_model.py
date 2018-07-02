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

        with tf.name_scope("dense1"):
            fc1 = tf.layers.dense(tf.layers.flatten(pool1), 128, name='dense1')
            fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)
            fc1 = tf.nn.relu(fc1)

        with tf.name_scope("dense2"):
            fc2 = tf.layers.dense(tf.layers.flatten(fc1), 128, name='dense2')
            fc2 = tf.contrib.layers.dropout(fc2, self.keep_prob)
            fc2 = tf.nn.relu(fc2)

        with tf.name_scope("softmax"):
            self.logits = tf.layers.dense(fc2, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.y_, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
