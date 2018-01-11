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


def generateTrig():
    img = np.zeros((512, 512, 3), np.uint8)
    pts = np.array([[np.random.randint(100, 500), np.random.randint(100, 500)],
                    [np.random.randint(100, 500), np.random.randint(100, 500)],
                    [np.random.randint(100, 500), np.random.randint(100, 500)]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    return x1 / 8, y1 / 8, x2 / 8, y2 / 8, img


# index, img = genRandom()
# winname = 'example'
# cv2.namedWindow(winname)
# cv2.imshow(winname, img)
# cv2.waitKey(0)
# cv2.destroyWindow(winname)

def genTF(train):
    if train:
        name = "train.tfrecords"
    else:
        name = "test.tfrecords"
    writer = tf.python_io.TFRecordWriter(temp_dir + name)
    for i in range(1000):
        x1, y1, x2, y2, img = generateTrig()
        img_raw = img.tostring()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "x1": tf.train.Feature(int64_list=tf.train.Int64List(value=[x1])),
            "y1": tf.train.Feature(int64_list=tf.train.Int64List(value=[y1])),
            "x2": tf.train.Feature(int64_list=tf.train.Int64List(value=[x2])),
            "y2": tf.train.Feature(int64_list=tf.train.Int64List(value=[y2])),
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
                                           'x1': tf.FixedLenFeature([], tf.int64),
                                           'y1': tf.FixedLenFeature([], tf.int64),
                                           'x2': tf.FixedLenFeature([], tf.int64),
                                           'y2': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    x1 = tf.cast(features['x1'], tf.int32)
    y1 = tf.cast(features['y1'], tf.int32)
    x2 = tf.cast(features['x2'], tf.int32)
    y2 = tf.cast(features['y2'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64])
    if normalize:
        img, [x1, y1, x2, y2] = preprocess(img, x1, y1, x2, y2)
    return img, [x1, y1, x2, y2]


def preprocess(img, x1, y1, x2, y2):
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    x1 = tf.cast(x1, tf.float32) * (1. / 64) - 0.5
    y1 = tf.cast(y1, tf.float32) * (1. / 64) - 0.5
    x2 = tf.cast(x2, tf.float32) * (1. / 64) - 0.5
    y2 = tf.cast(y2, tf.float32) * (1. / 64) - 0.5
    return img, [x1, y1, x2, y2]


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
        img = Image.fromarray(val[0])
        draw = ImageDraw.Draw(img)
        draw.rectangle(list(l[0]), outline=255)
        img.save("./temp/aa.bmp")

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    if not os.path.exists(temp_dir + "train.tfrecords"):
        genTF(train=True)
    showpic()
