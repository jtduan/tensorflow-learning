# coding=utf-8
import os

from cnn_model2 import *
from helper import *

if __name__ == '__main__':
    if not os.path.exists("./temp/train.tfrecords"):
        print "generating train data..."
        genTF(train=True)
    if not os.path.exists("./temp/test.tfrecords"):
        print "generating test data..."
        genTF(train=False)

    model = cnn_model()
    img, label = read_and_decode(temp_dir + "train.tfrecords")
    label_ = tf.reshape(label, [model.nums * model.classes])
    img_batch, label_batch = tf.train.shuffle_batch([img, label_],
                                                    batch_size=128, capacity=500,
                                                    min_after_dequeue=100)

    test_img, test_label = read_and_decode(temp_dir + "test.tfrecords")
    test_label = tf.reshape(test_label, [model.nums * model.classes])
    test_img_batch, test_label_batch = tf.train.shuffle_batch([test_img, test_label],
                                                              batch_size=128, capacity=500,
                                                              min_after_dequeue=100)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。

        saver = tf.train.Saver()
        # saver.restore(sess, "./temp/model2/")
        for epoch in range(1):
            x, y_ = sess.run([img_batch, label_batch])
            train_step, loss, temp = sess.run([model.train_step, model.loss, model.temp],
                                              feed_dict={model.x: x, model.y_: y_, model.keep_prob: 0.75})
            print temp.shape
            # if (epoch % 10 == 1):
            #     test_x, test_y_ = sess.run([test_img_batch, test_label_batch])
            #     y_pred_cls, acc = sess.run([model.y_pred_cls, model.acc],
            #                                feed_dict={model.x: test_x, model.y_: test_y_, model.keep_prob: 1})
            #     print("epoch %d ,acc=%f,loss=%f" % (epoch, acc, loss))
            #     if (acc > 0.95):
            #         builder = tf.saved_model.builder.SavedModelBuilder("./temp/model/outModel")
            #         inputs = {'input_x': tf.saved_model.utils.build_tensor_info(model.x),
            #                   'keep_prob': tf.saved_model.utils.build_tensor_info(model.keep_prob)}
            #         outputs = {'output': tf.saved_model.utils.build_tensor_info(model.y_pred_cls)}
            #         signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'classify_demo')
            #         builder.add_meta_graph_and_variables(sess, ['classify_demo_model'],
            #                                              signature_def_map={'signature': signature})
            #         builder.save()
            #         break
        # saver.save(sess, "./temp/model/")
        coord.request_stop()
        coord.join(threads)
