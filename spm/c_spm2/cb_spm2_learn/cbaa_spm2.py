# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os
import glob
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from scipy import signal
from datetime import datetime


print("######## debug ROI start ########")
print("######## debug ROI end   ########")

class SPM2Open_npz():  # second_spm.pyとして実装済み
    def unpack(self, files):
        print("===== npzファイルの解体 =====")
        print("読み込むフレーム数 : ", len(files))
        data_list_all_time = []
        label_list_all_time = []
        for file in files:
            data_per_pic, label_list_per_pic = self.load(file)
            data_list_all_time.append(data_per_pic)
            label_list_all_time.append(label_list_per_pic)
        data_list_all_time = np.array(data_list_all_time)
        label_list_all_time = np.array(label_list_all_time)

        print("===== windowごとに集計 =====")
        print("window数 : 6 (固定中。変更の場合はコード編集が必要）")
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

        print(f"画像加工の種類 : {win.shape[0]}種類")
        print(f"ヒストグラム特徴量の種類 : {win.shape[1]}種類")
        print(f"--- >>  合計 : {win.flatten().shape[0]}種類")
        print("===== 終了 =====")

        return self.data_list_all_win, self.label_list_all_win

    def load(self, file):
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
                labels=[]
                for feature in list(pic[f_key][w_key].keys()):
                    labels.append(f"{w_key}-{f_key}-{feature}")
                list_master_label[i].append(labels)
        list_master = np.array(list_master)
        return list_master, list_master_label


"""
    # 削除予定 (__init__を廃止したため、returnが可能になった)
    def get_data(self):
        return self.data_list_all_win,self.label_list_all_win
"""


class SPM2Learn():  # second_spm.pyとして実装済み
    """
    dataからmodelを作る。
    """

    def start(self, data_list_all_win, label_list_all_win, alpha=1.0, fps=30, stack_appear=23, stack_disappear=27, stack_info=None) -> None:
        self.fps = fps
        self.data_list_all_win = data_list_all_win
        self.label_list_all_win = label_list_all_win
        self.alpha = alpha
        # print(data_list_all_win.shape)#(win,pic_num,feature)=(6,886,30)
        if stack_info == None:
            self.stack_appear = stack_appear
            self.stack_disappear = stack_disappear
            self.stack_appear_frame = stack_appear*fps
            self.stack_disappear_frame = stack_disappear*fps
            self.stack_info = np.zeros((self.data_list_all_win.shape[0], 2))
            self.stack_info[:, 0] = int(self.stack_appear_frame)
            self.stack_info[:, 1] = int(self.stack_disappear_frame)
            # pprint(self.stack_info)
        else:
            self.stack_info = stack_info*self.fps
            pass
        self.initialize_model()
        self.fit()
        return self.model_master, self.label_list_all_win, self.scaler_master

    def initialize_model(self):
        self.model_master = []
        self.standardization_master = []
        self.scaler_master = []
        for i in range(self.data_list_all_win.shape[0]):
            self.model_master.append(Lasso(alpha=self.alpha, max_iter=100000))
            self.standardization_master.append(StandardScaler())
            self.scaler_master.append("")

    def fit(self):
        for win_no, win in enumerate(self.data_list_all_win):
            train_X = win
            self.scaler_master[win_no] = self.standardization_master[win_no].fit(
                train_X)
            train_X = self.scaler_master[win_no].transform(train_X)
            train_y = np.full((train_X.shape[0], 1), -100)
            # print(self.stack_info[win_no][0])
            train_y[int(self.stack_info[win_no][0]):int(
                self.stack_info[win_no][1])] = 100
            # print(train_X.shape, train_y.shape)
            self.model_master[win_no].fit(train_X, train_y)
            pass
        pass
    def get_nonzero_w(self):
        self.nonzero_w = []
        self.nonzero_w_label = []
        for win_no, (win_model, labels) in enumerate(zip(self.model_master, self.label_list_all_win)):
            self.nonzero_w.append([])
            self.nonzero_w_label.append([])
            weight = win_model.coef_
            labels = labels[0]
            for (w, label) in zip(weight, labels):
                if w > 1:
                    print("weight: \n", weight.shape)
                    print("labels: \n", labels.shape)
                    self.nonzero_w[win_no].append(w)
                    self.nonzero_w_label[win_no].append(label)
        self.nonzero_w_num = np.array([
            [len(self.nonzero_w_label[0]), len(
                self.nonzero_w_label[1]), len(self.nonzero_w_label[2])],
            [len(self.nonzero_w_label[3]), len(
                self.nonzero_w_label[4]), len(self.nonzero_w_label[5])]
        ])
        return self.nonzero_w, self.nonzero_w_label, self.nonzero_w_num

"""    
    def get_data(self):
        return self.model_master,self.label_list_all_win,self.scaler_master
"""


class SPM2Evaluate():  # 藤井さんの行動計画側に移設予定
    def start(self, model_master, test_data_list_all_win, test_label_list_all_win, scaler_master):
        self.model_master = model_master
        self.test_data_list_all_win = test_data_list_all_win
        self.test_label_list_all_win = test_label_list_all_win
        self.scaler_master = scaler_master
        if len(self.model_master) != len(self.test_data_list_all_win):
            print("学習済みモデルのウィンドウ数と、テストデータのウィンドウ数が一致しません")
            return None
        self.test()
        print(len(self.score_master))
        return self.score_master

    def test(self):
        # print(self.test_data_list_all_win)
        self.score_master = []
        for win_no in range(np.array(self.test_data_list_all_win).shape[0]):
            self.score_master.append([])
        for test_no in range(np.array(self.test_data_list_all_win).shape[1]):
            for win_no, win in enumerate(self.test_data_list_all_win):
                test_X = win[test_no]
                test_X = self.scaler_master[win_no].transform(
                    test_X.reshape(1, -1))
                score = self.model_master[win_no].predict(
                    test_X.reshape(1, -1))
                self.score_master[win_no].append(score)
                weight = self.model_master[win_no].coef_
    """        
    def get_score(self):
        return self.score_master
        # pprint(self.score_master[0])
    """

    def plot(self, save_dir):
        for i, win_score in enumerate(self.score_master):
            win_score = np.array(win_score).flatten()
            win_score_mov_ave = self.moving_average(win_score)
            win_score_low = self.lowpass(win_score, 25600, 100, 600, 3, 40)
            plt.plot(np.arange(len(win_score_mov_ave)),win_score_mov_ave, label=f"win_{i+1}", color="r")
            plt.plot(np.arange(len(win_score)),win_score_low, label=f"win_{i+1}")
        plt.xlabel("time")
        plt.ylabel("degree of risk")
        plt.ylim((-200, 200))
        global train_mov_code, test_mov_code, alpha
        plt.title(f"{train_mov_code} -->> {test_mov_code}  alpha={alpha}")
        plt.legend()
        name = str(datetime.now()).replace(" ", "").replace(
            ":", "").replace("-", "").replace(".", "")[:16]
        plt.savefig(save_dir+f"/cca{train_mov_code}{test_mov_code}_{name}.jpg")
        plt.cla()

        # plt.show()

    def moving_average(self, x, num=50):
        ave_data = np.convolve(x, np.ones(num)/num, mode="valid")
        return ave_data

    def lowpass(self, x, samplerate, fp, fs, gpass, gstop):
        fn = samplerate / 2  # ナイキスト周波数
        wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
        ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
        b, a = signal.butter(N, Wn, "low")  # フィルタ伝達関数の分子と分母を計算
        y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
        return y

    def get_nonzero_w(self):
        self.nonzero_w = []
        self.nonzero_w_label = []
        for win_no, (win_model, labels) in enumerate(zip(self.model_master, self.test_label_list_all_win)):
            self.nonzero_w.append([])
            self.nonzero_w_label.append([])
            weight = win_model.coef_
            labels = labels[0]
            for (w, label) in zip(weight, labels):
                if w > 1:
                    print("weight: \n", weight.shape)
                    print("labels: \n", labels.shape)
                    self.nonzero_w[win_no].append(w)
                    self.nonzero_w_label[win_no].append(label)
        self.nonzero_w_num = np.array([
            [len(self.nonzero_w_label[0]), len(
                self.nonzero_w_label[1]), len(self.nonzero_w_label[2])],
            [len(self.nonzero_w_label[3]), len(
                self.nonzero_w_label[4]), len(self.nonzero_w_label[5])]
        ])
        return self.nonzero_w, self.nonzero_w_label, self.nonzero_w_num


############  settings  #############
train_mov_codes = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                   'h', 'i', 'l', 'n', 'p', ]  # 'j','k','m','o',
test_mov_codes = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                  'h', 'i', 'l', 'n', 'p', ]  # 'j','k','m','o',
stack_starts = [0., 0., 9., 20., 11., 4., 35., 58., 10., 3., 6., 12., 16., ]
stack_ends = [4., 5., 16., 24., 13., 6., 36., 120., 11., 6., 8., 14., 19., ]
alpha = 5.0
spm_path = os.getcwd()

for train_mov_code, stack_start, stack_end in zip(train_mov_codes, stack_starts, stack_ends):
    ############   spm 2_1   ############
    train_dir_path = spm_path + \
        f"/b_spm1/b-data/bcca_secondinput/bcc{train_mov_code}"
    train_files = sorted(glob.glob(train_dir_path+"/*"))
    spm2_prep = SPM2Open_npz()
    train_datas, train_datas_label = spm2_prep.unpack(train_files)

    spm2_1 = SPM2Learn()
    model_master, _, scaler_master = spm2_1.start(
        train_datas, train_datas_label, alpha=alpha, stack_appear=stack_start, stack_disappear=stack_end)
    nonzero_w, nonzero_w_label, nonzero_w_num = spm2_1.get_nonzero_w()
    print(nonzero_w_num)


    for test_mov_code in test_mov_codes:
        ############   spm 2_2   ############
        test_dir_path = spm_path + \
            f"/b_spm1/b-data/bcca_secondinput/bcc{test_mov_code}"
        test_files = sorted(glob.glob(test_dir_path+"/*"))
        test_datas, test_datas_label = spm2_prep.unpack(test_files)
        spm2_2 = SPM2Evaluate()
        score_master = spm2_2.start(
            model_master, test_datas, test_datas_label, scaler_master)
        fig_dir_path = spm_path+"/c_spm2/cc_spm2_after/cca_output_of_spm2"
        spm2_2.plot(save_dir=fig_dir_path)
        nonzero_w, nonzero_w_label, nonzero_w_num = spm2_2.get_nonzero_w()

"""
# 単品モード
############  settings  #############
train_mov_codes = 'c'
test_mov_codes = 'x'
alpha = 5.0
for train_mov_code in train_mov_codes:
    for test_mov_code for test_mov_codes:

############ definitions ############
spm_path = os.getcwd()
train_dir_path = spm_path+f"/b_spm1/b-data/bcca_secondinput/bcc{train_mov_code}"
test_dir_path = spm_path+f"/b_spm1/b-data/bcca_secondinput/bcc{test_mov_code}"
fig_dir_path = spm_path+"/c_spm2/cc_spm2_after/cca_output_of_spm2"

############   spm 2_1   ############
train_files = sorted(glob.glob(train_dir_path+"/*"))
spm2_prep = SPM2Open_npz()
train_datas,train_datas_label=spm2_prep.unpack(train_files)

spm2_1 = SPM2Learn()
model_master, _, scaler_master = spm2_1.start(train_datas,train_datas_label,alpha=alpha,stack_appear=9,stack_disappear=16)

############   spm 2_2   ############
test_files = sorted(glob.glob(test_dir_path+"/*"))
test_datas,test_datas_label=spm2_prep.unpack(test_files)

spm2_2 = SPM2Evaluate()
score_master = spm2_2.start(model_master,test_datas,test_datas_label,scaler_master)
spm2_2.plot(save_dir=fig_dir_path)
nonzero_w,nonzero_w_label=spm2_2.get_nonzero_w()
nonzero_w_num=np.array([
    [len(nonzero_w_label[0]),len(nonzero_w_label[1]),len(nonzero_w_label[2])],
    [len(nonzero_w_label[3]),len(nonzero_w_label[4]),len(nonzero_w_label[5])]
    ])

print(nonzero_w_num)

"""
"""
メモ
・labelの表示関数
"""
