# coding=utf-8
import sys
import os

from helper import *

if __name__ == '__main__':
    if not os.path.exists("./temp/model/outModel/"):
        print "需要先运行训练程序run_cnn.py"
        sys.exit(0)
    sess = tf.InteractiveSession()
    signature_key = 'signature'
    input_key1 = 'input_x'
    input_key2 = 'keep_prob'
    output_key = 'output'

    meta_graph_def = tf.saved_model.loader.load(sess, ['classify_demo_model'], "./temp/model/outModel/")
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
    # img = generateRect()  # 2
    img = generateCircle()  # 1
    # img = generateTrig()  # 0
    img2 = preprocess(img)
    img3 = tf.expand_dims(img2, 0)
    img3 = sess.run(img3)
    # Image.fromarray(val[0]).save("./temp/aa.bmp")
    saver = tf.train.Saver()
    saver.save(sess,"./temp/model")

    print sess.run(y, feed_dict={x1: img3, x2: 1.0})  ##根据输入执行输出
    sess.close()
