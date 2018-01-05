# coding=utf-8
import tensorflow as tf

# # # 保存模型
# x1 = tf.placeholder(dtype=tf.int32)
# x2 = tf.placeholder(dtype=tf.int32, shape=[2, 3])
#
# v1 = tf.get_variable(name="v1", shape=[2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
# inc_v1 = v1.assign(v1 + 1)
#
# y_ = tf.placeholder(dtype=tf.int32)
#
# a = x1 + 1
# y = a * x2 + inc_v1
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# print sess.run(y, feed_dict={x1: 11, x2: [[1, 2, 3], [4, 5, 6]]})
# print sess.run(y, feed_dict={x1: 11, x2: [[1, 2, 3], [4, 5, 6]]})
#
# builder = tf.saved_model.builder.SavedModelBuilder("../model/testModel")
# inputs = {'x1': tf.saved_model.utils.build_tensor_info(x1),
#           'x2': tf.saved_model.utils.build_tensor_info(x2)}
# outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
# signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')
#
# outputs2 = {'output': tf.saved_model.utils.build_tensor_info(a)}
# signature2 = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs2, 'sig_name2')
#
# builder.add_meta_graph_and_variables(sess, ['test_saved_model'],
#                                      signature_def_map={'signature': signature, "signature2": signature2})
# builder.save()
# sess.close()


# 载入模型，输出最终结果或中间结果(前提已经起名字了)
sess = tf.InteractiveSession()
signature_key = 'signature'
input_key1 = 'x1'
input_key2 = 'x2'
output_key = 'output'

meta_graph_def = tf.saved_model.loader.load(sess, ['test_saved_model'], "../model/testModel")
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
print "获取原始图中的变量", sess.run(sess.graph.get_tensor_by_name("v1:0"))  ## 根据名字获取原始图中的变量
# _x 实际输入待inference的data
print sess.run(y, feed_dict={x1: 11, x2: [[1, 2, 3], [4, 5, 6]]}) ##根据输入执行输出
print sess.run(y, feed_dict={x1: 11, x2: [[1, 2, 3], [4, 5, 6]]})
sess.close()
