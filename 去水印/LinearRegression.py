# coding=utf-8
import pandas as pd

data = pd.read_csv('./submission.csv')
X_train = data['before']
X_train = X_train.reshape([-1, 1])
y_train = data['after']
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)
print linreg.intercept_  ## 截距
print linreg.coef_  ##斜率
