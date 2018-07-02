# coding=utf-8

'''
主要用于生成训练和测试数据
'''
import numpy as np
import os
import cv2


class DataIter():
    def __init__(self, pic_width=64):
        self.pic_width = pic_width

    def generateRamdomNumber(self, len):
        img = np.zeros((32, self.pic_width, 3), np.uint8)
        number = ['0', '1', '2', '3']
        text = ""
        label = np.zeros(len, dtype=np.int32)
        for i in range(len):
            label[i] = number[np.random.randint(0, 4)]
            text = text + str(label[i])

        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, text,
        #             (np.random.randint(5, 20), np.random.randint(24, 30)), font, 0.8 + np.random.randn() / 7,
        #             (255, 255, 255), 2)
        cv2.putText(img, text,
                    (0, 26), font, 0.7,
                    (255, 255, 255), 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img, label

    def sparse_tuple_from_label(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int32)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)

        return indices, values, shape

    def next_batch(self, batch_size):
        images = []
        labels = []
        for _ in range(batch_size):
            img, label = self.generateRamdomNumber(np.random.randint(2, 4))
            images.append(np.transpose(img))
            labels.append(label)
        labels_spase = self.sparse_tuple_from_label(labels)
        images = np.array(images, dtype=np.float32)
        return images, labels_spase, [self.pic_width] * batch_size

# img ,label = DataIter().generateRamdomNumber(9)
#
# print label
# winname = 'example'
# cv2.namedWindow(winname)
# cv2.imshow(winname, img)
# cv2.waitKey(0)
# cv2.destroyWindow(winname)
