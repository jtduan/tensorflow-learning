# coding=utf-8

import tensorflow as tf


class cnn_lstm_ctc_model(object):
    pic_height = 32
    learning_rate = 1e-3
    DECAY_STEPS = 5000
    LEARNING_RATE_DECAY_FACTOR = 0.9
    num_hidden = 64
    num_layers = 2
    num_classes = 4

    def __init__(self, pic_width=64, lstm_len=64):
        self.lstm_len = lstm_len
        self.pic_width = pic_width  # n_features
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, None, self.pic_height])  # (batch_size, pic_width, pic_height)
        self.y_ = tf.sparse_placeholder(dtype=tf.int32)
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.ctc()

    def ctc(self):
        batch_size = tf.shape(self.x)[0]
        input = tf.reshape(self.x, shape=[batch_size, self.pic_width, -1, 1])  # or input = tf.expand_dims(self.x, -1)
        with tf.name_scope("cnn1"):
            w = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[32]), name="B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            conv = tf.layers.batch_normalization(conv)
            conv = tf.nn.relu(conv + b)  # (batch_size,pic_width,pic_height=32,channel=32)
            conv = tf.nn.max_pool(conv, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
            # lstm_input = tf.reshape(conv, [-1, self.pic_width, self.pic_height * 16])

            conv_layer_flat = tf.reshape(conv, [-1, self.pic_width, self.pic_height * 16])  # ->
            dense1 = tf.layers.dense(conv_layer_flat, 32 * 4, name='dense')
            lstm_input = tf.reshape(dense1, [batch_size, self.lstm_len, 32 * 4])
            self.temp = lstm_input

        with tf.name_scope("lstm"):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(cell, lstm_input, sequence_length=self.seq_len,
                                           dtype=tf.float32, time_major=False)  # [batch_size,max_time_step,num_hidden]
            outputs = tf.reshape(outputs, [-1, self.num_hidden])  # [batch_size*max_time_step,num_hidden]

            outputs_dense = tf.layers.dense(outputs, self.num_classes + 1)  # [batch_size*max_time_step,11]

            y = tf.reshape(outputs_dense, [-1, self.lstm_len, self.num_classes + 1])
            self.y = tf.transpose(y, (1, 0, 2))  # [max_timesteps,batch_size,num_classes]

            # self.y = tf.reshape(outputs_dense,
            #                     [self.pic_width, -1, self.num_classes + 1])  # [max_timesteps,batch_size,num_classes]

            self.global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       self.global_step,
                                                       self.DECAY_STEPS,
                                                       self.LEARNING_RATE_DECAY_FACTOR,
                                                       staircase=True)

        with tf.name_scope("optimize"):
            self.loss = tf.nn.ctc_loss(labels=self.y_, inputs=self.y, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,
                                                                                           global_step=self.global_step)

            # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.cost,
            #                                                                               global_step=self.global_step)

        with tf.name_scope("accuracy"):
            # self.decoded, log_prob = tf.nn.ctc_greedy_decoder(self.y, self.seq_len)
            self.decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.y, self.seq_len, merge_repeated=False)
            self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.y_))
            # # max = tf.maximum(decoded[0].dense_shape[1], self.y_.dense_shape[1])
            # #
            self.pred = tf.sparse_to_dense(self.decoded[0].indices, self.decoded[0].dense_shape,
                                           self.decoded[0].values, default_value=-1)
