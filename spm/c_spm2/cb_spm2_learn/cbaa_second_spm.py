# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os

path=os.getcwd()
train_X=np.load(path+"/second_input_data/2022-06-09_18-44-40.npz")["array_1"]

# 特徴画像の数＊特徴量ベクトル＊学習画像の数　を想定
print(train_X.shape)
train_y=np.zeros((train_X.shape[0],1))
train_y[-5:]=1
model=Lasso()
model.fit(train_X,train_y)
test_X=train_X
test_y=model.predict(test_X)
print(model.score(test_X,test_y))

"""
・Xを1次元にも3次元にもできない問題
・今後何らかの支障になるかもなと思い、できればここをロバストにしてみたいと思うところです。
"""