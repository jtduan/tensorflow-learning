这是一个识别不定长数字的问题
cnn+lstm+ctc模型中,一般不在cnn和lstm之间添加全连接层（添加之后模型难以理解）
cnn可以使用channel当做lstm输入的max_time_step,或者使用
卷积层后直接跟lstm层(图片宽度128，lstm_len=128)：1000轮,时间10分钟(0.6s/batch),准确率 98%
卷积层后跟全连接层(图片宽度64，lstm_len=64)再跟lstm层：1500轮(0.2s/batch) 准确率95%(实际效果不好)