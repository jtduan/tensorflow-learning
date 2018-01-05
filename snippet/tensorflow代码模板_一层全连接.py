# coding=utf-8
import tensorflow as tf
import numpy as np
import pandas as pd

arr = np.asarray([1, 1, 1, 1, 2, 1, 0, 1, 3, 0, 1, 0, 4, 0, 0, 0]).reshape(4, 4)
train_data = pd.DataFrame(arr, columns=["id", "age", "sex", "out"])
target = train_data['out']
train_data = train_data.drop("id", 1)
train_data = train_data.drop("out", 1)
train_data = tf.convert_to_tensor(np.array(train_data).tolist(), dtype=tf.float32)
dataset1 = tf.data.Dataset.from_tensor_slices((train_data, target))
dataset1 = dataset1.shuffle(buffer_size=1000)
dataset1 = dataset1.repeat(1000)
dataset1 = dataset1.batch(3)
iterator = dataset1.make_one_shot_iterator()
next_x, next_y = iterator.get_next()

#
# training_df = pd.read_csv("./Titanic/resources/train.csv")
# testing_df = pd.read_csv("./Titanic/resources/test.csv")
#
#
# def drop_useless_cols(df):
#     return df.drop(["Name", "Ticket", "Cabin"], axis=1)
#
#
# training_df = drop_useless_cols(training_df)
# testing_df = drop_useless_cols(testing_df)
#
#
# def create_pclass_cols(df):
#     # Creates new column for each pclass
#     df["pclass_1"] = df["Pclass"].apply(lambda x: 1 if x == 1 else 0)
#     df["pclass_2"] = df["Pclass"].apply(lambda x: 1 if x == 2 else 0)
#     df["pclass_3"] = df["Pclass"].apply(lambda x: 1 if x == 3 else 0)
#     return df.drop("Pclass", axis=1)
#
#
# training_df = create_pclass_cols(training_df)
# testing_df = create_pclass_cols(testing_df)
#
#
# def fill_age(df):
#     fill_value = df["Age"].mode()[0]
#     return df.fillna(fill_value)
#
#
# training_df = fill_age(training_df)
# testing_df = fill_age(testing_df)
#
#
# def one_hot_encoding(df):
#     df["Embarked"] = df["Embarked"].replace(24.0, "S")
#     return pd.get_dummies(df)
#
#
# training_df = one_hot_encoding(training_df)
# testing_df = one_hot_encoding(testing_df)
#
# features = training_df.drop(["PassengerId", "Survived"], axis=1).values
# labels = training_df["Survived"].values

##############深度学习模型,本例为1层全连接模型

class DEMO_MODEL(object):

    def inference(self, inputs, out_units):
        W = tf.Variable(tf.random_normal([2, out_units], stddev=2))
        b = tf.Variable(tf.random_normal([out_units], stddev=2))
        y = tf.matmul(inputs, W) + b
        return y

    #######定义损失函数
    def loss(self, logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    ######训练函数
    def training(self, loss, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 2])
        self.y_ = tf.placeholder(tf.float32)

        self.out = self.inference(self.x, 2)  # 模型的one-hot输出
        self.losses = self.loss(self.out, [self.y_])
        self.train_op = self.training(self.losses, 0.1)  # 模型训练步骤

        self.y_pred = tf.argmax(tf.nn.softmax(self.out), 1)  # 模型实际输出

        correct_pred = tf.equal(tf.argmax(self.y_, 1), self.y_pred)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # 模型准确率


def start():
    model = DEMO_MODEL()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for step in xrange(1000):
        try:
            input_x, input_y = sess.run([next_x, next_y])
            input_y = np.eye(3, 2)[input_y]
            sess.run(model.train_op, feed_dict={model.x: input_x, model.y_: input_y})
        except tf.errors.OutOfRangeError:
            pass

        print sess.run(model.y_pred, feed_dict={model.x: [[1., 0.]], model.y_: [[0, 1]]})

    sess.close()


start()
