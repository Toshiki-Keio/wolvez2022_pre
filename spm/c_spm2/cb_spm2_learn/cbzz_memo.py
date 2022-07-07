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

    def get_data(self):
        return self.data_list_all_win,self.label_list_all_win

class Learn():
    """
    dataからmodelを作る。
    """
    def __init__(self,data_list_all_win,label_list_all_win,fps=30,stack_appear=23,stack_disappear=27,stack_info=None) -> None:
        self.fps = fps
        self.data_list_all_win=data_list_all_win
        self.label_list_all_win=label_list_all_win
        # print(data_list_all_win.shape)#(win,pic_num,feature)=(6,886,30)
        if stack_info==None:
            self.stack_appear = stack_appear
            self.stack_disappear = stack_disappear
            self.stack_appear_frame = stack_appear*fps
            self.stack_disappear_frame = stack_disappear*fps
            self.stack_info=np.zeros((self.data_list_all_win.shape[0],2))
            self.stack_info[:,0]=int(self.stack_appear_frame)
            self.stack_info[:,1]=int(self.stack_disappear_frame)
            # pprint(self.stack_info)
        else:
            self.stack_info=stack_info*self.fps
            pass
        self.initialize_model()
        self.fit()


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
            # print(self.stack_info[win_no][0])
            train_y[int(self.stack_info[win_no][0]):int(self.stack_info[win_no][1])] = 100
            # print(train_X.shape, train_y.shape)
            self.model_master[win_no].fit(train_X, train_y)
            pass
        pass
    
    def get_data(self):
        return self.model_master,self.label_list_all_win,self.scaler_master


class Evaluate():
    def __init__(self,model_master,test_data_list_all_win,scaler_master,train_code,test_code):
        self.model_master=model_master
        self.test_data_list_all_win=test_data_list_all_win
        self.scaler_master=scaler_master
        self.train_code=train_code
        self.test_code=test_code
        if len(self.model_master)!=len(self.test_data_list_all_win):
            print("学習済みモデルのウィンドウ数と、テストデータのウィンドウ数が一致しません")
            return None
        self.test()
        self.plot()


    def test(self):
        # print(self.test_data_list_all_win)
        self.score_master=[]
        for win_no in range(np.array(self.test_data_list_all_win).shape[0]):
            self.score_master.append([])
        for test_no in range(np.array(self.test_data_list_all_win).shape[1]):
            for win_no, win in enumerate(self.test_data_list_all_win):
                test_X = win[test_no]
                # print(f"test_X win_no\n: {win_no}",test_X)
                test_X=self.scaler_master[win_no].transform(test_X.reshape(1, -1))
                score = self.model_master[win_no].predict(test_X.reshape(1, -1))
                # print(score)
                self.score_master[win_no].append(score)
                pass
        # pprint(self.score_master[0])
    
    def plot(self):
        for i, win_score in enumerate(self.score_master):
            plt.plot(np.arange(len(win_score)), win_score, label=f"win_{i+1}")
        plt.xlabel("time")
        plt.ylabel("degree of risk")
        plt.legend()
        plt.savefig(f"c_spm2/cc_spm2_after/cca_output_of_spm2/cca{self.train_code}{self.test_code}_L-bcc{self.train_code}_P-bcc{self.test_code}.png")
        plt.cla()
        # plt.show()




# wolvez2022/spmで実行してください
train_codes=['a','b','c','d','e','f','g','h','i']
test_codes=['a','b','c','d','e','f','g','h','i']
stack_starts=[0.,0.,9.,20.,11.,4.,35.,58.,10.,]#bそもそもスタートがスタック,g白砂利道,hスタック以外の原因で修理・パソコン映り込みも,iスタック以外の原因で停止は学習データ自体が悪い
stack_ends=[4.,5.,16.,24.,13.,6.,36.,120.,11.,]
# train_codes=['a','c','d','e','f']
# test_codes=['a','c','d','e','f']
# stack_starts=[0.,9.,20.,11.,4.]#bそもそもスタートがスタック,g白砂利道,hスタック以外の原因で修理・パソコン映り込みも,iスタック以外の原因で停止は学習データ自体が悪い
# stack_ends=[4.,16.,24.,13.,6.]

for train_code in train_codes:
    print("train data mov code : ",train_code)
    spm_path = os.getcwd()
    train_files = sorted(glob.glob(spm_path+f"/b_spm1/b-data/bcca_secondinput/bcc{train_code}/*"))

    seq1=Open_npz(train_files)
    data_list_all_win,label_list_all_win=seq1.get_data()

    """
    stack_info=np.array([[12., 18.],
        [12., 18.],
        [12., 18.],
        [12., 18.],
        [12., 18.],
        [12, 18.]])
    「stackした」と学習させるフレームの指定方法
    1. 全ウィンドウで一斉にラベリングする場合
        Learnの引数でstack_appearおよびstack_disappearを[s]で指定する。
    2. ウィンドウごとに個別にラベリングする場合
    stack_info=np.array(
        [
            [win_1_stack_start,win_1_stack_end],
            [win_2_stack_start,win_2_stack_end],
            ...
            [win_6_stack_start,win_6_stack_end],
        ]
    )
    t[s]で入力すること。
    """
    seq2=Learn(data_list_all_win,label_list_all_win,fps=30,stack_appear=5,stack_disappear=6,stack_info=None)
    #seq2=Learn(data_list_all_win,label_list_all_win,fps=30,stack_info=stack_info)
    model_master,label_list_all_win,scaler_master=seq2.get_data()

    spm_path = os.getcwd()
    for test_code in test_codes:
        print('test data mov code : ',test_code)
        test_files = sorted(glob.glob(spm_path+f"/b_spm1/b-data/bcca_secondinput/bcc{test_code}/*"))

        seq3=Open_npz(test_files)
        test_data_list_all_win,test_label_list_all_win=seq3.get_data()

        seq4=Evaluate(model_master,test_data_list_all_win,scaler_master,train_code,test_code)
        del seq3
        del seq4
