# coding=utf-8

'''
主要用于生成训练和测试数据
'''
import numpy as np
import os
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw

temp_dir = "./temp/"


def generateRamdomNumber(len):
    img = np.zeros((32, 128, 3), np.uint8)
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    text = ""
    label = np.zeros(len, dtype=np.int32)
    for i in range(len):
        label[i] = number[np.random.randint(0, 10)]
        text = text + str(label[i])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text,
                (np.random.randint(5, 20), np.random.randint(24, 30)), font, 0.8 + np.random.randn() / 7,
                (255, 255, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img, label


# img, label = generateRamdomNumber(5)
# print label
#    winname = 'example'
#     cv2.namedWindow(winname)
#     cv2.imshow(winname, img)
#     cv2.waitKey(0)
#     cv2.destroyWindow(winname)


def genTF(train):
    if train:
        name = "train.tfrecords"
    else:
        name = "test.tfrecords"
    writer = tf.python_io.TFRecordWriter(temp_dir + name)
    for i in range(5000):
        img, label = generateRamdomNumber(5)
        img_raw = img.tostring()
        label_raw = label.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_and_decode(filename, normalize=True):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label_raw': tf.FixedLenFeature([], tf.string),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    label = tf.decode_raw(features['label_raw'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [32, 128])
    label = tf.reshape(label, [5])
    if normalize:
        img = preprocess(img)
    label = tf.one_hot(tf.cast(label, tf.int32), 10, axis=1)
    return img, label


def preprocess(img):
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img


def showpic():
    img, label = read_and_decode(temp_dir + "train.tfrecords", normalize=False)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=1, capacity=500,
                                                    min_after_dequeue=100)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
        val, l = sess.run([img_batch, label_batch])
        Image.fromarray(val[0]).save("./temp/aa.bmp")
        print(val[0].shape, l[0])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    if not os.path.exists(temp_dir + "train.tfrecords"):
        genTF(train=True)
    showpic()
    # generateRamdomNumber(5)
