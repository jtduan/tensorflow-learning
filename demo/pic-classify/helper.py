# coding=utf-8

'''
主要用于生成训练和测试数据
'''
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

temp_dir = "./temp/"


def generateRect():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.rectangle(img, (np.random.randint(10, 500), np.random.randint(10, 500)),
                  (np.random.randint(10, 500), np.random.randint(10, 500)),
                  (255, 255, 255), thickness=-1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    return img


def generateCircle():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.ellipse(img, center=(np.random.randint(200, 400), np.random.randint(200, 400)),
                axes=(np.random.randint(30, 100), np.random.randint(30, 100)), angle=0, startAngle=0, endAngle=360,
                color=(255, 255, 255),
                thickness=-1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    return img


def generateTrig():
    img = np.zeros((512, 512, 3), np.uint8)
    pts = np.array([[np.random.randint(100, 500), np.random.randint(100, 500)],
                    [np.random.randint(100, 500), np.random.randint(100, 500)],
                    [np.random.randint(100, 500), np.random.randint(100, 500)]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    return img


def genRandom():
    index = np.random.randint(0, 3)
    if (index == 0):
        return 0, generateTrig()
    if (index == 1):
        return 1, generateCircle()
    return 2, generateRect()


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
        index, img = genRandom()
        img_raw = img.tostring()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [48, 48])
    img = preprocess(img)
    label = tf.one_hot(tf.cast(features['label'], tf.int32), 3)
    return img, label


def preprocess(img):
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img


def showpic():
    img, label = read_and_decode(temp_dir + "train.tfrecords")
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
        print(val.shape, l)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    showpic()
