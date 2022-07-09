# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os
import glob
from pprint import pprint
from sklearn.preprocessing import StandardScaler

class SPM2Open_npz(): # second_spm.pyとして実装済み
    def unpack(self,files):
        # self.data_list_all_win,self.label_list_all_win=self.unpack(files)
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

class SPM2Learn():# second_spm.pyとして実装済み
    """
    dataからmodelを作る。
    """
    def start(self,data_list_all_win,label_list_all_win,fps=30,stack_appear=23,stack_disappear=27,stack_info=None) -> None:
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


class SPM2Evaluate(): # 藤井さんの行動計画側に移設予定
    def __init__(self,model_master,test_data_list_all_win,test_label_list_all_win,scaler_master):
        self.model_master=model_master
        self.test_data_list_all_win=test_data_list_all_win
        self.test_label_list_all_win=test_label_list_all_win
        self.scaler_master=scaler_master
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
                weight=self.model_master[win_no].coef_
                
                pass
        # pprint(self.score_master[0])
        
    def get_score(self):
        return self.score_master
    
    def plot(self):
        for i, win_score in enumerate(self.score_master):
            plt.plot(np.arange(len(win_score)), win_score, label=f"win_{i+1}")
        plt.xlabel("time")
        plt.ylabel("degree of risk")
        plt.title(f"Learn from mov bcc{self.train_code}, Predict mov bcc{self.test_code}")
        plt.legend()
        # plt.savefig(f"c_spm2/cc_spm2_after/cca_output_of_spm2/cca{self.train_code}{self.test_code}_L-bcc{self.train_code}_P-bcc{self.test_code}.png")
        plt.cla()
        
        # plt.show()
