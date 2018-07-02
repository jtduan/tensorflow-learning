# coding=utf-8
import tensorflow as tf
import time

from helper import *
from cnn_lstm_ctc_model import *

batch_size = 64
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


def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))


if __name__ == '__main__':
    it = DataIter(128, 128)
    model = cnn_lstm_ctc_model(128, 128)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        # saver.restore(sess, "./temp/")
        for epoch in range(2):
            start = time.time()
            img_batch, label_batch, len_batch = it.next_batch(batch_size)
            temp = sess.run(model.temp, feed_dict={model.x: img_batch, model.y_: label_batch,
                                                   model.seq_len: len_batch})
            print temp.shape
        #     _, cost, steps = sess.run([model.train_step, model.cost, model.global_step],
        #                               feed_dict={model.x: img_batch, model.y_: label_batch,
        #                                          model.seq_len: len_batch})
        #
        #     seconds = time.time() - start
        #     print("Step:", steps, ", batch seconds:", seconds, ", cost:", cost)
        #     if (epoch % 10 == 0):
        #         img_batch2, label_batch2, len_batch2 = it.next_batch(batch_size)
        #
        #         acc, decoded, pred = sess.run([model.acc, model.decoded[0], model.pred],
        #                                       feed_dict={model.x: img_batch2, model.y_: label_batch2,
        #                                                  model.seq_len: len_batch2})
        #         report_accuracy(decoded, label_batch2)
        #         # saver.save(sess, "./temp/")
        #
        # builder = tf.saved_model.builder.SavedModelBuilder("./temp/model/outModel")
        # inputs = {'input_x': tf.saved_model.utils.build_tensor_info(model.x),
        #           'seq_len': tf.saved_model.utils.build_tensor_info(model.seq_len)}
        # outputs = {'output': tf.saved_model.utils.build_tensor_info(model.pred)}
        # signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'ctc_demo')
        # builder.add_meta_graph_and_variables(sess, ['ctc_demo_model'],
        #                                      signature_def_map={'signature': signature})
        #
        # builder.save()
