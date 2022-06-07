# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

train_X=np.random.rand(10,5)# 特徴画像の数＊特徴量ベクトル＊学習画像の数　を想定
train_y=np.random.rand(10)

model=Lasso()
model.fit(train_X,train_y)

"""
・Xを3次元にできない問題
・今後何らかの支障になるかもなと思い、できればここをロバストにしてみたいと思うところです。
"""