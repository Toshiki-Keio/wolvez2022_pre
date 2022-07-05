# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os
import glob
from pprint import pprint
from sklearn.preprocessing import StandardScaler




class Open_npz():
    def __init__(self,files):
        self.data_list_all_win,self.label_list_all_win=self.unpack(files)
        # とりあえず片っ端からとってくる

    def unpack(self,files):
        data_list_all_time = []
        label_list_all_time = []
        for file in files:
            data_per_pic, label_list_per_pic=self.load(file)
            data_list_all_time.append(data_per_pic)
            label_list_all_time.append(label_list_per_pic)
        data_list_all_time = np.array(data_list_all_time)
        label_list_all_time = np.array(label_list_all_time)

        # windowごとにまとめる
        self.data_list_all_win = [[], [], [], [], [], []]
        self.label_list_all_win = [[], [], [], [], [], []]
        for pic, lab_pic in zip(data_list_all_time, label_list_all_time):
            for win_no, (win, label_win) in enumerate(zip(pic, lab_pic)):
                self.data_list_all_win[win_no].append(win.flatten())
                self.label_list_all_win[win_no].append(label_win.flatten())
                # print(train_X.shape)
                pass
        self.data_list_all_win = np.array(self.data_list_all_win)
        self.label_list_all_win = np.array(self.label_list_all_win)
        return self.data_list_all_win,self.label_list_all_win
    
    def load(self,file):
        pic = np.load(file, allow_pickle=True)['array_1'][0]
        # pprint(pic)
        feature_keys = list(pic.keys())
        list_master = [[], [], [], [], [], []]
        list_master_label = [[], [], [], [], [], []]
        for f_key in feature_keys:
            window_keys = list(pic[f_key].keys())
            for i, w_key in enumerate(window_keys):
                # print(list(pic[f_key][w_key].values()))
                list_master[i].append(list(pic[f_key][w_key].values()))
                labels = [f"{w_key}-{f_key}-{list(pic[f_key][w_key].keys())[0]}", f"{w_key}-{f_key}-{list(pic[f_key][w_key].keys())[1]}",
                        f"{w_key}-{f_key}-{list(pic[f_key][w_key].keys())[2]}"]
                list_master_label[i].append(labels)
        list_master = np.array(list_master)
        return list_master, list_master_label

    def get_train_X(self):
        return self.data_list_all_win,self.label_list_all_win

class Learn():
    """
    dataからmodelを作る。
    """
    def __init__(self,data_list_all_win,label_list_all_win,fps=30,stack_appear=23,stack_disappear=27) -> None:
        self.fps = fps
        self.stack_appear = stack_appear
        self.stack_disappear = stack_disappear
        self.stack_appear_frame = stack_appear*fps
        self.stack_disappear_frame = stack_disappear*fps

        self.data_list_all_win=data_list_all_win
        self.label_list_all_win=label_list_all_win
        print(data_list_all_win.shape)#(win,pic_num,feature)=(6,886,30)
        self.initialize_model()
        self.fit()

        pass

    def initialize_model(self):
        self.model_master=[]
        self.standardization_master=[]
        self.scaler_master=[]
        for i in range(self.data_list_all_win.shape[0]):
            self.model_master.append(Lasso(max_iter=100000))
            self.standardization_master.append(StandardScaler())
            self.scaler_master.append("")
    
    def fit(self):
        for win_no, win in enumerate(self.data_list_all_win):
            train_X = win
            self.scaler_master[win_no]=self.standardization_master[win_no].fit(train_X)
            train_X=self.scaler_master[win_no].transform(train_X)
            train_y = np.zeros((train_X.shape[0], 1))
            train_y[self.stack_appear_frame:self.stack_disappear_frame] = 1
            print(train_X.shape, train_y.shape)
            self.model_master[win_no].fit(train_X, train_y)
            pass
        pass
        





# wolvez2022/spmで実行してください
spm_path = os.getcwd()
train_files = sorted(glob.glob(spm_path+"/b_spm1/b-data/bcca_secondinput/*"))

train=Open_npz(train_files)
data_list_all_win,label_list_all_win=train.get_train_X()

Learn(data_list_all_win,label_list_all_win)
