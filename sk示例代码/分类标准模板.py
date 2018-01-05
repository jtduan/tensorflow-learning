# coding=utf-8
'''
这是一个集合各类机器学习算法的标准分类模板，
使用本模板之前，可能需要手动进行数据预处理
示例代码以https://www.kaggle.com/c/leaf-classification数据集为示例
'''
# coding=utf-8

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def warn(*args, **kwargs): pass


import warnings

warnings.warn = warn

from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('./resources/train.csv')
test = pd.read_csv('./resources/test.csv')


def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings
    classes = list(le.classes_)  # save column names for submission
    test_ids = test.id  # save test ids for submission

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


'''
数据与处理，将训练数据格式化，分成训练数据和测试数据
'''
train, labels, test, test_ids, classes = encode(train, test)

sss = StratifiedShuffleSplit(test_size=0.2, random_state=23)

for train_index, test_index in sss.split(train, labels):
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

'''
PCA降维优化：显示PCA不同参数时的准确度提升效果
'''
# n_components_array=([1,2,3,4,5,10,20,50,100])
# vr = np.zeros(len(n_components_array))
# i=0
# for n_components in n_components_array:
#     pca = PCA(n_components=n_components)
#     pca.fit(train)
#     vr[i] = sum(pca.explained_variance_ratio_)
#     i=i+1
# plt.figure(figsize=(8,4))
# plt.plot(n_components_array,vr,'k.-')
# plt.xscale("log")
# plt.ylim(9e-2,1.1)
# plt.yticks(np.linspace(0.2,1.0,9))
# plt.xlim(0.9)
# plt.grid(which="both")
# plt.xlabel("number of PCA components",size=20)
# plt.ylabel("variance ratio",size=20)
# plt.show()
#
# pca = PCA(n_components=50)
# pca.fit(train)
# train = pca.transform(train)
# test = pca.transform(test)


'''
评估各类模型效果
'''
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    print("=" * 30)
    print(name)

    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print ("Accuracy: {:.4%}".format(clf.score(X_test, y_test)))
    # print("Accuracy: {:.4%}".format(acc))

    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))

print("=" * 30)

'''
选择一个好的模型之后，需要进行该模型的调参
'''
favorite_clf = RandomForestClassifier()

parameters = {"n_estimators": [40, 100, 150],
              "max_features": [8, 5, 3]}

grid = GridSearchCV(favorite_clf, parameters)

grid.fit(X_train, y_train)
print("随机森林最优参数： %s 最优分数 %s"
      % (grid.best_params_, grid.best_score_))

'''
使用优化后的参数在测试集上查看结果
'''
favorite_clf = RandomForestClassifier(n_estimators=150, max_features=8)
favorite_clf.fit(X_train, y_train)
print "模型在测试集上的准确率：", favorite_clf.score(X_test, y_test)

'''
预测结果并生成csv提交
'''
test_predictions = favorite_clf.predict_proba(test)

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

submission.to_csv('submission.csv', index=False)
# print submission.tail()
