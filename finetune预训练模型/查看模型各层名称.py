# coding=utf-8
import numpy as np
import tensorflow as tf
import download
import os

########################################################################
# 压缩包地址.
data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# 数据保存地址.
data_dir = "temp/"

# ImageNet 各个分类的名称. (Downloaded)
path_uid_to_cls = "imagenet_2012_challenge_label_map_proto.pbtxt"

# File containing the mappings between uid and string. (Downloaded)
path_uid_to_name = "imagenet_synset_to_human_label_map.txt"

# 网络层定义. (Downloaded)
path_graph_def = "classify_image_graph_def.pb"


########################################################################


def maybe_download():
    """
   如果inception-v3模型不存在就下载，大概85M.
    """

    print("Downloading Inception v3 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


class Inception:
    """
    预训练好的inception-v3包含1000种分类.
    """
    # 数据层.
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"

    # resize后的数据.
    tensor_name_resized_image = "ResizeBilinear:0"
    # softmax层的名字.
    tensor_name_softmax_logits = "softmax/logits:0"

    # 最后一层的池化.
    tensor_name_transfer_layer = "pool_3:0"

    def __init__(self):

        # 创建tensorflow计算图.
        self.graph = tf.Graph()

        # 将新的计算图设置为默认图.
        with self.graph.as_default():
            # 打开pre_trained模型.
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # 复制定义好的计算图到新的图中，先创建一个空的图.
                graph_def = tf.GraphDef()

                # 加载proto-buf中的模型.
                graph_def.ParseFromString(file.read())

                # 最后复制pre-def图的到默认图中.
                tf.import_graph_def(graph_def, name='')

                # 完成从proto-buf的加载.

        # 获取最后softmax层特征数据.
        self.y_logits = self.graph.get_tensor_by_name(self.tensor_name_softmax_logits)

        # 获取计算图最后一层的数据,可以更改对应名称.
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

        # 获取最后一层的长度.
        self.transfer_len = self.transfer_layer.get_shape()[3]

        # 创建会话执行图.
        self.session = tf.Session(graph=self.graph)

    def close(self):
        """
        关闭会话.
        """

        self.session.close()

    def _create_feed_dict(self, image_path=None, image=None):
        """
        """
        if image is not None:
            # Image is passed in as a 3-dim array that is already decoded.
            feed_dict = {self.tensor_name_input_image: image}

        elif image_path is not None:
            # Read the jpeg-image as an array of bytes.
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            # Image is passed in as a jpeg-encoded image.
            feed_dict = {self.tensor_name_input_jpeg: image_data}

        else:
            raise ValueError("Either image or image_path must be set.")

        return feed_dict

    def transfer_values(self, image_path=None, image=None):
        """
        计算对应层数据

        :param image_path:
            输入图像路径.

        :param image:
            输入图像数据.

        :return:
            对应层数据.
        """

        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        # 变成一维数据输出
        transfer_values = np.squeeze(transfer_values)

        return transfer_values


model = Inception()
# 查看模型各层的名字
names = [op.name for op in model.graph.get_operations()]

# values = model.transfer_values(image_path="./temp/cropped_panda.jpg")

print names
# print values
