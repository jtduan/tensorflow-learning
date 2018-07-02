# coding=utf-8

import tensorflow as tf


class ctc_model(object):
    pic_height = 32
    learning_rate = 1e-3
    DECAY_STEPS = 5000
    LEARNING_RATE_DECAY_FACTOR = 0.9
    num_hidden = 64
    num_layers = 2

    def __init__(self, pic_width=64):
        self.pic_width = pic_width
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 32])
        self.y_ = tf.sparse_placeholder(dtype=tf.int32)
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.ctc()

    def ctc(self):
        with tf.name_scope("lstm"):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
            # cell2 = tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True)
            # cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(cell, self.x, sequence_length=self.seq_len,
                                           dtype=tf.float32)  # [batch_size,max_time_step,num_hidden]
            outputs = tf.reshape(outputs, [-1, self.num_hidden])  # [batch_size*max_time_step,num_hidden]

            outputs_dense = tf.layers.dense(outputs, 5)  # [batch_size*max_time_step,12]
            # W = tf.Variable(tf.truncated_normal([self.num_hidden, 12],
            #                                     stddev=0.1), name="W")
            # b = tf.Variable(tf.constant(0., shape=[12]), name="b")
            #
            # outputs_dense = tf.matmul(outputs, W) + b

            y = tf.reshape(outputs_dense, [-1, self.pic_width, 5])
            self.y = tf.transpose(y, (1, 0, 2))  # [max_timesteps,batch_size,num_classes]
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

            # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.cost)

        with tf.name_scope("accuracy"):
            # self.decoded, log_prob = tf.nn.ctc_greedy_decoder(self.pred, self.seqLen)
            self.decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.y, self.seq_len, merge_repeated=False)
            self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.y_))
            # # max = tf.maximum(decoded[0].dense_shape[1], self.y_.dense_shape[1])
            # #
            self.pred = tf.sparse_to_dense(self.decoded[0].indices, self.decoded[0].dense_shape,
                                           self.decoded[0].values, default_value=-1)
            # self.actual = tf.sparse_to_dense(self.y_.indices, self.y_.dense_shape, self.y_.values, default_value=-1)
            # self.actual_contact = tf.sparse_to_dense(self.y_.indices, (decoded[0].dense_shape[0], max),
            #                                          self.y_.values, default_value=-1)
            #
            # correct_pred = tf.equal(tf.cast(self.pred, tf.int32), self.actual_contact)
            # self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
