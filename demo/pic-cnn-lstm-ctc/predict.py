# coding=utf-8
import sys
import tensorflow as tf
import os

from helper import *

DIGITS = '0123456789'


def decode_sparse_tensor(sparse_tensor):
    # print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    # print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        # print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        # print(result)
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    # str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    # str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return decoded


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    signature_key = 'signature'
    input_key1 = 'input_x'
    input_key2 = 'seq_len'
    output_key = 'output'

    meta_graph_def = tf.saved_model.loader.load(sess, ['ctc_demo_model'], "./temp/model/outModel/")
    # 从meta_graph_def中取出SignatureDef对象
    signature = meta_graph_def.signature_def

    # 从signature中找出具体输入输出的tensor name
    x_tensor_name1 = signature[signature_key].inputs[input_key1].name
    x_tensor_name2 = signature[signature_key].inputs[input_key2].name
    y_tensor_name = signature[signature_key].outputs[output_key].name

    # 获取tensor 并inference
    x1 = sess.graph.get_tensor_by_name(x_tensor_name1)
    x2 = sess.graph.get_tensor_by_name(x_tensor_name2)
    y = sess.graph.get_tensor_by_name(y_tensor_name)
    # print "获取原始图中的变量", sess.run(sess.graph.get_tensor_by_name("v1:0"))
    # _x 实际输入待inference的data

    img, label = DataIter(128,128).generateRamdomNumber(7)
    img = np.transpose(img)
    img3 = tf.expand_dims(img, 0)
    img3 = sess.run(img3)
    print label
    # Image.fromarray(val[0]).save("./temp/aa.bmp")

    out = sess.run(y, feed_dict={x1: img3, x2: [128]})  ##根据输入执行输出
    print out
    sess.close()
