# coding=utf-8

import pandas as pd
import numpy as np
import tensorflow as tf

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

## 训练集和验证集大小
_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

def myfn():
    print "ok"

def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have either run data_download.py or '
            'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    # labels = tf.Print(labels, [labels], message="训练数据=") ## 加入此行可以输出训练数据的label
    return features, labels


train = pd.read_csv("./data/census_data/adult.data", header=None)

relationships = np.unique(train[7].values)
educations = np.unique(train[3].values)
marital_statuses = np.unique(train[5].values)
workclasses = np.unique(train[1].values)

#转换为one-hot数据格式
relationship = tf.feature_column.categorical_column_with_vocabulary_list('relationship', relationships)
marital_status = tf.feature_column.categorical_column_with_vocabulary_list('marital_status', marital_statuses)
education = tf.feature_column.categorical_column_with_vocabulary_list('education', educations)
workclass = tf.feature_column.categorical_column_with_vocabulary_list('workclass', workclasses)


occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)

age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]
crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]

model_dir = "../model"
model = tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=base_columns + crossed_columns,
                                      optimizer=tf.train.FtrlOptimizer(
                                          learning_rate=0.1,
                                          l1_regularization_strength=1.0,
                                          l2_regularization_strength=1.0)
                                      )

model.train(input_fn=lambda: input_fn("./data/census_data/adult.data", num_epochs=50, shuffle=True, batch_size=30))
results = model.evaluate(input_fn=lambda: input_fn("./data/census_data/adult.test", 1, shuffle=False, batch_size=30))
for key in sorted(results):
    print('%s: %s' % (key, results[key]))
