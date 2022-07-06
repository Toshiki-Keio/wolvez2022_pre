# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os
import glob
import pickle
from pprint import pprint

files=glob.glob("spm/b_spm1/b-data/bcca_secondinput/*")
train_X=[]
for file in files:
    data_dicts=np.load(file,allow_pickle=True)["array_1"][0].values()
    train_X_situ=[]
    for dict in data_dicts:
        element=list(dict.values())
        for ele in element:
            train_X_situ.append(ele)
    train_X.append(train_X_situ)
train_X=np.array(train_X)

"""
for file in files:
    pprint(np.load(file,allow_pickle=True)["array_1"][0])
path=os.getcwd()
"""#train_X=np.load(path+"/second_input_data/2022-06-09_18-44-40.npz")["array_1"]
# 特徴画像の数＊特徴量ベクトル＊学習画像の数　を想定
test_X=train_X[-1].reshape(1,-1)
train_X=train_X[:-1]
train_y=np.zeros((train_X.shape[0],1))
test_y=0
print(train_X.shape,test_X.shape,train_y.shape)
train_y[-1:]=1
model=Lasso(max_iter=1000)
model.fit(train_X,train_y)
possibility=model.predict(test_X)
print(possibility)
"""
・Xを1次元にも3次元にもできない問題
・今後何らかの支障になるかもなと思い、できればここをロバストにしてみたいと思うところです。
"""