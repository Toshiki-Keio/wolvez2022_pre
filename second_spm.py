# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

train_X=np.random.rand(10,5)
# 特徴画像の数＊特徴量ベクトル＊学習画像の数　を想定
print(train_X.shape)
train_y=np.random.rand(10)
#test_X=np.random.rand(10,5).reshape(-1,1)

model=Lasso()
model.fit(train_X,train_y)

#test_y=model.predict(test_X)


"""
・Xを1次元にも3次元にもできない問題
・今後何らかの支障になるかもなと思い、できればここをロバストにしてみたいと思うところです。
"""