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
            train_y = np.full((train_X.shape[0], 1),-100)
            # print(self.stack_info[win_no][0])
            train_y[int(self.stack_info[win_no][0]):int(self.stack_info[win_no][1])] = 100
            # print(train_X.shape, train_y.shape)
            self.model_master[win_no].fit(train_X, train_y)
            pass
        pass
    
    def get_data(self):
        return self.model_master,self.label_list_all_win,self.scaler_master


class SPM2Evaluate(): # 藤井さんの行動計画側に移設予定
    def start(self,model_master,test_data_list_all_win,test_label_list_all_win,scaler_master,train_code,test_code):
        self.model_master=model_master
        self.test_data_list_all_win=test_data_list_all_win
        self.test_label_list_all_win=test_label_list_all_win
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
                weight=self.model_master[win_no].coef_
                
    def get_score(self):
        return self.score_master
        # pprint(self.score_master[0])
    
    def plot(self):
        for i, win_score in enumerate(self.score_master):
            plt.plot(np.arange(len(win_score)), win_score, label=f"win_{i+1}")
        plt.xlabel("time")
        plt.ylabel("degree of risk")
        plt.title(f"Learn from mov bcc{self.train_code}, Predict mov bcc{self.test_code}")
        plt.legend()
        plt.savefig(f"c_spm2/cc_spm2_after/ccb_-100_100/ccb{self.train_code}{self.test_code}_L-bcc{self.train_code}_P-bcc{self.test_code}.png")
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

for train_code,stack_start,stack_end in zip(train_codes,stack_starts,stack_ends):
    print("train data mov code : ",train_code)
    spm_path = os.getcwd()
    seq1=SPM2Open_npz()
    train_files = sorted(glob.glob(spm_path+f"/b_spm1/b-data/bcca_secondinput/bcc{train_code}/*"))
    print(f"{len(train_files)} frames found from mov code {train_code}")
    seq1.unpack(train_files)
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
    seq2=SPM2Learn()
    seq2.start(data_list_all_win,label_list_all_win,fps=30,stack_appear=stack_start,stack_disappear=stack_end,stack_info=None)
    #seq2=Learn(data_list_all_win,label_list_all_win,fps=30,stack_info=stack_info)
    model_master,label_list_all_win,scaler_master=seq2.get_data()

    spm_path = os.getcwd()
    for test_code in test_codes:
        print('test data mov code : ',test_code)
        test_files = sorted(glob.glob(spm_path+f"/b_spm1/b-data/bcca_secondinput/bcc{test_code}/*"))

        seq3=SPM2Open_npz()
        seq3.unpack(test_files)
        test_data_list_all_win,test_label_list_all_win=seq3.get_data()

        seq4=SPM2Evaluate()
        seq4.start(model_master,test_data_list_all_win,test_label_list_all_win,scaler_master,train_code,test_code)
        del seq3
        del seq4




"""
メモ
・stack時刻がミスしていた

・重み表示を復活させる
・区間平均の機能を入れる
"""







# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Lasso
# import os
# import glob
# import pickle
# from pprint import pprint
# from sklearn.preprocessing import StandardScaler



# def unpack(npz_path):
#     """
#     元画像1枚に関するデータが全部入ったnpz_pathを渡すと、listへの復元と、listのどこに何が入っているかの案内の発行をしてくれる
#     """
#     pic = np.load(npz_path, allow_pickle=True)['array_1'][0]
#     # pprint(pic)
#     feature_keys = list(pic.keys())
#     list_master = [[], [], [], [], [], []]
#     list_master_label = [[], [], [], [], [], []]
#     for f_key in feature_keys:
#         window_keys = list(pic[f_key].keys())
#         for i, w_key in enumerate(window_keys):
#             # print(list(pic[f_key][w_key].values()))
#             list_master[i].append(list(pic[f_key][w_key].values()))
#             labels = [f"{w_key}-{f_key}-{list(pic[f_key][w_key].keys())[0]}", f"{w_key}-{f_key}-{list(pic[f_key][w_key].keys())[1]}",
#                       f"{w_key}-{f_key}-{list(pic[f_key][w_key].keys())[2]}"]
#             list_master_label[i].append(labels)
#     list_master = np.array(list_master)
#     return list_master, list_master_label


# # get data
# spm_path = os.getcwd()
# # print(spm_path)
# print(spm_path+"/b_spm1/b-data/bcca_secondinput/*")
# files = sorted(glob.glob(spm_path+"/b_spm1/b-data/bcca_secondinput/*"))
# # pprint(files)

# # npzを解凍。
# data_list_all_time = []
# label_list_all_time = []
# for file in files:
#     data_per_pic, label_list_per_pic = unpack(file)
#     data_list_all_time.append(data_per_pic)
#     label_list_all_time.append(label_list_per_pic)
# data_list_all_time = np.array(data_list_all_time)
# label_list_all_time = np.array(label_list_all_time)
# # print(label_list_all_time.shape)#撮影した写真の枚数、ウィンドウの数、特徴画像の種類、特徴画像の特徴量の種類
# # print("data_list_all_time")
# # pprint(label_list_all_time)

# # ウィンドウごとに整理
# data_list_all_win = [[], [], [], [], [], []]
# label_list_all_win = [[], [], [], [], [], []]
# for pic, lab_pic in zip(data_list_all_time, label_list_all_time):
#     for win_no, (win, label_win) in enumerate(zip(pic, lab_pic)):
#         data_list_all_win[win_no].append(win.flatten())
#         label_list_all_win[win_no].append(label_win.flatten())
#         # print(train_X.shape)
#         pass
# data_list_all_win = np.array(data_list_all_win)
# label_list_all_win = np.array(label_list_all_win)

# fps = 30
# stack_appear = 23
# stack_disapper = 27
# stack_appear_frame = stack_appear*fps
# stack_disappear_frame = stack_disapper*fps
# total_frame = len(files)


# # train
# test_num = 50
# model_master = [Lasso(max_iter=100000), Lasso(max_iter=100000), Lasso(
#     max_iter=100000), Lasso(max_iter=100000), Lasso(max_iter=100000), Lasso(max_iter=100000)]
# standardization_list=[StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler()]
# scaler_list=["","","","","",""]#####
# for win_no, win in enumerate(data_list_all_win):
#     train_X = win[:-test_num]
#     scaler_list[win_no]=standardization_list[win_no].fit(train_X)
#     train_X=scaler_list[win_no].transform(train_X)
#     train_y = np.zeros((train_X.shape[0], 1))
#     train_y[stack_appear_frame:stack_disappear_frame] = 1
#     print(train_X.shape, train_y.shape)
#     model_master[win_no].fit(train_X, train_y)


# # test
# score_master = [[], [], [], [], [], []]
# for test_no in range(test_num):
#     for win_no, win in enumerate(data_list_all_win):
#         test_X = win[-test_no]
#         print(f"test_X win_no: {win_no}",test_X.shape)
#         test_X=scaler_list[win_no].transform(test_X.reshape(1, -1))
#         print(f"test_X win_no: {win_no}",test_X.shape)
#         score = model_master[win_no].predict(test_X.reshape(1, -1))
#         score_master[win_no].append(score)
#         pass
# pprint(label_list_all_win)

# for i, win_score in enumerate(score_master):
#     plt.plot(np.arange(len(win_score)), win_score, label=f"win_{i}")

# plt.xlabel("time")
# plt.ylabel("degree of risk")
# plt.legend()
# plt.show()

# valuable_weight=[]
# valuable_label=[]
# for i,model in enumerate(model_master):
#     print(f"win_{i+1}重み係数: ",model_master[i].coef_.shape)
#     weight = model_master[i].coef_
#     print(weight)
#     label_list=label_list_all_win[i][0]
#     for j,w in enumerate(weight):
#         if w >=1e-6:
#             valuable_weight.append(w)
#             valuable_label.append(label_list[j])

# pprint(valuable_label)
# pprint(len(valuable_label))

# print(files)
# """
# npzの中身
# array(
#     [
#         {
#             'normalRGB': 
#             {
#                 'win_1': {'var': 198.5714006259781, 'med': 242.0, 'ave': 8349.509847177527}, 
#                 'win_2': {'var': 138.55516431924883, 'med': 231.0, 'ave': 13560.616909949526}, 
#                 'win_3': {'var': 164.68951486697966, 'med': 240.0, 'ave': 12550.98544561754}, 
#                 'win_4': {'var': 185.2595461658842, 'med': 237.0, 'ave': 9515.470976945833}, 
#                 'win_5': {'var': 188.0658841940532, 'med': 236.0, 'ave': 8919.395346283929}, 
#                 'win_6': {'var': 214.89233176838812, 'med': 236.0, 'ave': 4407.548423201353}
#             }, 
#             'enphasis': 
#             {
#                 'win_1': {'var': 144.23415492957747, 'med': 139.0, 'ave': 1054.3623467428201}, 
#                 'win_2': {'var': 147.34287949921753, 'med': 141.0, 'ave': 2041.7802427256984}, 
#                 'win_3': {'var': 148.43485915492957, 'med': 142.0, 'ave': 1708.3702480631046}, 
#                 'win_4': {'var': 147.291744913928, 'med': 142.0, 'ave': 2110.4347988332097}, 
#                 'win_5': {'var': 145.8273082942097, 'med': 142.0, 'ave': 2391.6417738188825}, 
#                 'win_6': {'var': 147.90790297339592, 'med': 142.0, 'ave': 1627.7706261189114}
#             }, 
#             'edge': 
#             {
#                 'win_1': {'var': 127.77691705790298, 'med': 137.0, 'ave': 699.0108756284884}, 'win_2': {'var': 128.13270735524256, 'med': 132.0, 'ave': 777.6750335152981}, 'win_3': {'var': 128.0511345852895, 'med': 133.0, 'ave': 750.8916340804784}, 'win_4': {'var': 128.2794992175274, 'med': 130.0, 'ave': 619.540737777386}, 'win_5': {'var': 127.95156494522692, 'med': 131.0, 'ave': 784.3300484116663}, 'win_6': {'var': 128.1994522691706, 'med': 130.0, 'ave': 657.8328322508515}}, 'hsv': {'win_1': {'var': 187.78748043818467, 'med': 200.0, 'ave': 3148.890046703452}, 'win_2': {'var': 201.05066510172145, 'med': 215.0, 'ave': 2933.7095222493454}, 'win_3': {'var': 215.22535211267606, 'med': 224.0, 'ave': 1541.848433952699}, 'win_4': {'var': 217.35966353677622, 'med': 227.0, 'ave': 1705.3529191356192}, 'win_5': {'var': 223.4076682316119, 'med': 232.0, 'ave': 1364.1589239838263}, 'win_6': {'var': 222.2679186228482, 'med': 226.0, 'ave': 656.0300975458034}}, 'red': {'win_1': {'var': 161.20528169014085, 'med': 170.0, 'ave': 1151.2600894746415}, 'win_2': {'var': 161.3956572769953, 'med': 168.0, 'ave': 808.351319168926}, 'win_3': {'var': 160.36874021909233, 'med': 163.0, 'ave': 507.72471140199735}, 'win_4': {'var': 159.6904538341158, 'med': 161.0, 'ave': 664.9101279630486}, 'win_5': {'var': 158.12719092331767, 'med': 160.0, 'ave': 760.0379304502462}, 'win_6': {'var': 155.25426447574335, 'med': 157.0, 'ave': 757.7365624089257}}, 'blue': {'win_1': {'var': 138.91220657276995, 'med': 151.0, 'ave': 1586.6412750997379}, 'win_2': {'var': 138.76862284820032, 'med': 149.0, 'ave': 1246.1235381660508}, 'win_3': {'var': 137.6596635367762, 'med': 141.0, 'ave': 613.2421131888269}, 'win_4': {'var': 136.65336463223787, 'med': 138.0, 'ave': 806.932895565009}, 'win_5': {'var': 136.46784037558686, 'med': 139.0, 'ave': 1107.6047560558973}, 'win_6': {'var': 136.49096244131457, 'med': 139.0, 'ave': 939.0661155056316}}, 'green': {'win_1': {'var': 200.95062597809076, 'med': 205.0, 'ave': 426.81203794930946}, 'win_2': {'var': 206.75387323943661, 'med': 211.0, 'ave': 449.41207419989416}, 'win_3': {'var': 205.92214397496087, 'med': 208.0, 'ave': 376.0210902390522}, 'win_4': {'var': 201.28501564945228, 'med': 203.0, 'ave': 515.0839069246377}, 'win_5': {'var': 201.65007824726135, 'med': 204.0, 'ave': 467.57403364019973}, 'win_6': {'var': 199.75762910798122, 'med': 201.0, 'ave': 391.34020001267385}}, 'purple': {'win_1': {'var': 172.99949139280125, 'med': 181.0, 'ave': 960.1941703203483}, 'win_2': {'var': 173.02910798122065, 'med': 180.0, 'ave': 761.807446935132}, 'win_3': {'var': 170.90958528951487, 'med': 174.0, 'ave': 408.04741985931406}, 'win_4': {'var': 169.1106807511737, 'med': 170.0, 'ave': 485.44592661013024}, 'win_5': {'var': 167.56412363067292, 'med': 170.0, 'ave': 682.9769523227436}, 'win_6': {'var': 164.56905320813772, 'med': 167.0, 'ave': 545.7310689001423}}, 'emerald': {'win_1': {'var': 211.76036776212834, 'med': 215.0, 'ave': 423.85450909792786}, 'win_2': {'var': 217.54573552425666, 'med': 221.0, 'ave': 478.95002093787724}, 'win_3': {'var': 216.45989827856025, 'med': 218.0, 'ave': 311.79408825256354}, 'win_4': {'var': 210.14115805946793, 'med': 212.0, 'ave': 559.7842215070987}, 'win_5': {'var': 210.58348982785603, 'med': 213.0, 'ave': 545.3585224063911}, 'win_6': {'var': 209.0109546165884, 'med': 211.0, 'ave': 384.8271100433237}}, 'yellow': {'win_1': {'var': 223.54859154929576, 'med': 235.0, 'ave': 2460.552803180586}, 'win_2': {'var': 204.16694053208138, 'med': 239.0, 'ave': 6859.470683284411}, 'win_3': {'var': 218.91451486697966, 'med': 239.0, 'ave': 4210.439914514254}, 'win_4': {'var': 215.16694053208138, 'med': 233.0, 'ave': 3839.3939227210326}, 'win_5': {'var': 214.97781690140846, 'med': 232.0, 'ave': 3599.095321681655}, 'win_6': {'var': 222.97132237871674, 'med': 229.0, 'ave': 1209.0354451996714}}}],
#       dtype=object)
#       array([{'normalRGB': {'win_1': {'var': 215.56224569640062, 'med': 230.0, 'ave': 3195.8999595890855}, 'win_2': {'var': 214.76670579029735, 'med': 235.0, 'ave': 4219.097803858667}, 'win_3': {'var': 226.50293427230048, 'med': 233.0, 'ave': 1429.5986611866033}, 'win_4': {'var': 219.1612676056338, 'med': 228.0, 'ave': 1806.7597517578083}, 'win_5': {'var': 217.53399843505477, 'med': 227.0, 'ave': 2055.5625373771495}, 'win_6': {'var': 219.7568075117371, 'med': 223.0, 'ave': 729.1737212634176}}, 'enphasis': {'win_1': {'var': 143.5997261345853, 'med': 138.0, 'ave': 990.6911501597394}, 'win_2': {'var': 147.90598591549295, 'med': 142.0, 'ave': 1925.6460518057486}, 'win_3': {'var': 148.95266040688577, 'med': 142.0, 'ave': 1562.1574616233306}, 'win_4': {'var': 147.44460093896714, 'med': 142.0, 'ave': 2051.6545991756484}, 'win_5': {'var': 146.5946791862285, 'med': 142.0, 'ave': 2287.076716602869}, 'win_6': {'var': 147.6079420970266, 'med': 142.0, 'ave': 1535.5973469387445}}, 'edge': {'win_1': {'var': 127.84847417840376, 'med': 136.0, 'ave': 616.3121338221032}, 'win_2': {'var': 127.79389671361503, 'med': 134.0, 'ave': 793.3504791818202}, 'win_3': {'var': 128.07793427230047, 'med': 134.0, 'ave': 714.0031594260397}, 'win_4': {'var': 128.1819640062598, 'med': 131.0, 'ave': 675.9225620268735}, 'win_5': {'var': 128.03732394366196, 'med': 131.0, 'ave': 837.9326444819148}, 'win_6': {'var': 128.16897496087637, 'med': 131.0, 'ave': 706.4433175721429}}, 'hsv': {'win_1': {'var': 192.83525039123631, 'med': 199.0, 'ave': 2228.7354944991193}, 'win_2': {'var': 204.66651017214397, 'med': 213.0, 'ave': 2056.9484871951236}, 'win_3': {'var': 213.92668231611893, 'med': 220.0, 'ave': 1162.2954069898437}, 'win_4': {'var': 218.45340375586855, 'med': 223.0, 'ave': 542.7318663487182}, 'win_5': {'var': 223.5823552425665, 'med': 228.0, 'ave': 595.7526072853833}, 'win_6': {'var': 216.56302816901407, 'med': 222.0, 'ave': 970.4468099225242}}, 'red': {'win_1': {'var': 157.41017214397496, 'med': 167.0, 'ave': 1093.2516336166889}, 'win_2': {'var': 157.00262128325508, 'med': 164.0, 'ave': 761.1137255232404}, 'win_3': {'var': 155.60254303599373, 'med': 159.0, 'ave': 493.16561951105876}, 'win_4': {'var': 154.6814945226917, 'med': 156.0, 'ave': 653.0255886897196}, 'win_5': {'var': 153.70586854460095, 'med': 156.0, 'ave': 763.5706854349887}, 'win_6': {'var': 151.40446009389672, 'med': 154.0, 'ave': 671.1127813595186}}, 'blue': {'win_1': {'var': 137.87906885758997, 'med': 151.0, 'ave': 1583.7699218246794}, 'win_2': {'var': 137.20915492957747, 'med': 147.0, 'ave': 1156.638492087108}, 'win_3': {'var': 136.09945226917057, 'med': 140.0, 'ave': 522.4988729394277}, 'win_4': {'var': 135.22257433489827, 'med': 137.0, 'ave': 900.1781993195917}, 'win_5': {'var': 135.195813771518, 'med': 139.0, 'ave': 990.6757805975569}, 'win_6': {'var': 135.16725352112675, 'med': 138.0, 'ave': 903.1276991861183}}, 'green': {'win_1': {'var': 192.45473395931143, 'med': 196.0, 'ave': 400.5174345536355}, 'win_2': {'var': 196.7258998435055, 'med': 201.0, 'ave': 402.7528033765102}, 'win_3': {'var': 195.16924882629107, 'med': 197.0, 'ave': 302.7416208754877}, 'win_4': {'var': 191.19835680751174, 'med': 193.0, 'ave': 520.714801681765}, 'win_5': {'var': 191.75262128325508, 'med': 194.0, 'ave': 444.86599469381923}, 'win_6': {'var': 189.8014475743349, 'med': 192.0, 'ave': 409.7599900798024}}, 'purple': {'win_1': {'var': 167.08928012519561, 'med': 175.0, 'ave': 745.9538443957081}, 'win_2': {'var': 166.91424100156493, 'med': 173.0, 'ave': 641.6664325616365}, 'win_3': {'var': 164.54612676056337, 'med': 167.0, 'ave': 374.1069490042761}, 'win_4': {'var': 162.76267605633802, 'med': 164.0, 'ave': 580.8207508981905}, 'win_5': {'var': 161.9111111111111, 'med': 165.0, 'ave': 681.6852130064337}, 'win_6': {'var': 159.381338028169, 'med': 162.0, 'ave': 560.1467174585069}}, 'emerald': {'win_1': {'var': 202.4273082942097, 'med': 205.0, 'ave': 305.02092874846016}, 'win_2': {'var': 206.5368544600939, 'med': 209.0, 'ave': 356.3450423747492}, 'win_3': {'var': 204.00782472613457, 'med': 206.0, 'ave': 286.49860857021804}, 'win_4': {'var': 199.3059076682316, 'med': 201.0, 'ave': 453.54597448912745}, 'win_5': {'var': 200.00324726134585, 'med': 202.0, 'ave': 466.3752241970779}, 'win_6': {'var': 197.87668231611894, 'med': 200.0, 'ave': 355.2573279601098}}, 'yellow': {'win_1': {'var': 220.59741784037558, 'med': 223.0, 'ave': 596.9005254138289}, 'win_2': {'var': 222.00888106416275, 'med': 228.0, 'ave': 1181.0719477307682}, 'win_3': {'var': 222.85023474178405, 'med': 225.0, 'ave': 439.1132511186052}, 'win_4': {'var': 217.02230046948358, 'med': 220.0, 'ave': 705.2464510458683}, 'win_5': {'var': 216.02930359937403, 'med': 220.0, 'ave': 873.378523145699}, 'win_6': {'var': 213.51729264475745, 'med': 216.0, 'ave': 386.10533476725425}}}],
#       dtype=object)
# array([{'normalRGB': {'win_1': {'var': 216.064593114241, 'med': 231.0, 'ave': 3266.313941970594}, 'win_2': {'var': 213.85438184663536, 'med': 236.0, 'ave': 4618.9937405803275}, 'win_3': {'var': 226.24354460093898, 'med': 234.0, 'ave': 1666.1983933825961}, 'win_4': {'var': 218.7234350547731, 'med': 229.0, 'ave': 2074.816821690227}, 'win_5': {'var': 217.29053208137717, 'med': 228.0, 'ave': 2251.427875929722}, 'win_6': {'var': 220.29823943661972, 'med': 224.0, 'ave': 784.5237684184134}}, 'enphasis': {'win_1': {'var': 143.55567292644758, 'med': 138.0, 'ave': 993.2981524814423}, 'win_2': {'var': 147.1203834115806, 'med': 141.0, 'ave': 1956.6029178498657}, 'win_3': {'var': 148.73806729264476, 'med': 142.0, 'ave': 1565.0444976730928}, 'win_4': {'var': 147.27363067292646, 'med': 142.0, 'ave': 2059.1588508244736}, 'win_5': {'var': 146.41216744913928, 'med': 142.0, 'ave': 2307.666620341288}, 'win_6': {'var': 147.525, 'med': 141.0, 'ave': 1549.9240228873239}}, 'edge': {'win_1': {'var': 127.83114241001564, 'med': 136.0, 'ave': 613.4054464257288}, 'win_2': {'var': 127.91478873239437, 'med': 133.0, 'ave': 744.7711427957415}, 'win_3': {'var': 128.1322378716745, 'med': 134.0, 'ave': 646.5814959308975}, 'win_4': {'var': 128.16686228482004, 'med': 131.0, 'ave': 706.727829904352}, 'win_5': {'var': 127.92492175273865, 'med': 131.0, 'ave': 808.8915072317492}, 'win_6': {'var': 128.19287949921753, 'med': 131.0, 'ave': 687.1043467945562}}, 'hsv': {'win_1': {'var': 191.37582159624412, 'med': 198.0, 'ave': 2369.949368456435}, 'win_2': {'var': 203.80974178403756, 'med': 213.0, 'ave': 2162.3611805280034}, 'win_3': {'var': 213.19072769953053, 'med': 219.0, 'ave': 1271.913270831956}, 'win_4': {'var': 218.12187010954617, 'med': 223.0, 'ave': 565.4072525277294}, 'win_5': {'var': 223.77132237871675, 'med': 229.0, 'ave': 550.0950852622691}, 'win_6': {'var': 217.79010172143975, 'med': 221.0, 'ave': 683.798704840983}}, 'red': {'win_1': {'var': 157.60512519561814, 'med': 167.0, 'ave': 1082.6549893818221}, 'win_2': {'var': 157.27554773082943, 'med': 164.0, 'ave': 750.9407009910708}, 'win_3': {'var': 155.89687010954617, 'med': 159.0, 'ave': 495.5669855089501}, 'win_4': {'var': 155.0404538341158, 'med': 156.0, 'ave': 664.8826357877749}, 'win_5': {'var': 153.98708920187792, 'med': 156.0, 'ave': 780.0553106195861}, 'win_6': {'var': 151.53372456964007, 'med': 154.0, 'ave': 662.6168595235122}}, 'blue': {'win_1': {'var': 137.94260563380283, 'med': 151.0, 'ave': 1567.04791871928}, 'win_2': {'var': 137.40168231611892, 'med': 147.0, 'ave': 1171.840724869343}, 'win_3': {'var': 136.19503129890452, 'med': 140.0, 'ave': 525.4908751555149}, 'win_4': {'var': 135.2888888888889, 'med': 137.0, 'ave': 832.2696731003305}, 'win_5': {'var': 135.29808294209704, 'med': 139.0, 'ave': 936.4267568882692}, 'win_6': {'var': 135.26259780907668, 'med': 138.0, 'ave': 866.2993522498231}}, 'green': {'win_1': {'var': 192.91572769953052, 'med': 197.0, 'ave': 408.4004099164628}, 'win_2': {'var': 197.33419405320814, 'med': 201.0, 'ave': 410.3044332706375}, 'win_3': {'var': 195.7213615023474, 'med': 198.0, 'ave': 302.04575651876837}, 'win_4': {'var': 191.64600938967135, 'med': 193.0, 'ave': 537.0894793801936}, 'win_5': {'var': 192.35, 'med': 195.0, 'ave': 472.46693661971824}, 'win_6': {'var': 190.31627543035995, 'med': 192.0, 'ave': 404.47618268470154}}, 'purple': {'win_1': {'var': 168.31815336463225, 'med': 176.0, 'ave': 854.0843026932243}, 'win_2': {'var': 167.30770735524257, 'med': 174.0, 'ave': 646.1090329284435}, 'win_3': {'var': 164.92007042253522, 'med': 168.0, 'ave': 373.34717151303755}, 'win_4': {'var': 163.2677621283255, 'med': 165.0, 'ave': 566.6152361499899}, 'win_5': {'var': 162.17136150234742, 'med': 165.0, 'ave': 672.6991954859044}, 'win_6': {'var': 159.73791079812207, 'med': 162.0, 'ave': 573.142068248694}}, 'emerald': {'win_1': {'var': 203.0577856025039, 'med': 206.0, 'ave': 312.6841021386972}, 'win_2': {'var': 207.16205007824726, 'med': 210.0, 'ave': 358.4283563605594}, 'win_3': {'var': 204.72621283255086, 'med': 207.0, 'ave': 287.58004841166627}, 'win_4': {'var': 199.91819248826292, 'med': 201.0, 'ave': 449.890138516939}, 'win_5': {'var': 200.55712050078247, 'med': 203.0, 'ave': 469.3891472640399}, 'win_6': {'var': 198.48333333333332, 'med': 200.0, 'ave': 362.0745266040689}}, 'yellow': {'win_1': {'var': 220.56729264475743, 'med': 224.0, 'ave': 730.6659724824342}, 'win_2': {'var': 222.34323161189357, 'med': 229.0, 'ave': 1283.412669368891}, 'win_3': {'var': 223.45582942097028, 'med': 226.0, 'ave': 478.4904589755977}, 'win_4': {'var': 217.72233959311424, 'med': 221.0, 'ave': 745.912380441797}, 'win_5': {'var': 216.64851330203442, 'med': 221.0, 'ave': 894.6642505283834}, 'win_6': {'var': 214.21584507042255, 'med': 216.0, 'ave': 404.9683952561221}}}],
#       dtype=object)
# array([{'normalRGB': {'win_1': {'var': 211.05156494522691, 'med': 240.0, 'ave': 5860.957200211353}, 'win_2': {'var': 168.05575117370893, 'med': 238.0, 'ave': 11821.62596066422}, 'win_3': {'var': 195.8328247261346, 'med': 241.0, 'ave': 8776.658007044396}, 'win_4': {'var': 202.01862284820032, 'med': 235.0, 'ave': 6685.003409058069}, 'win_5': {'var': 202.12331768388106, 'med': 234.0, 'ave': 6458.592852216761}, 'win_6': {'var': 220.41557120500784, 'med': 233.0, 'ave': 2677.185516536009}}, 'enphasis': {'win_1': {'var': 144.52609546165885, 'med': 139.0, 'ave': 1103.6349215307932}, 'win_2': {'var': 147.32046165884194, 'med': 141.0, 'ave': 2068.703916218796}, 'win_3': {'var': 148.57797339593114, 'med': 142.0, 'ave': 1748.974280086929}, 'win_4': {'var': 146.99424882629108, 'med': 142.0, 'ave': 2146.0560310867554}, 'win_5': {'var': 145.54675273865413, 'med': 141.0, 'ave': 2397.1409284224296}, 'win_6': {'var': 147.84397496087638, 'med': 142.0, 'ave': 1640.76947465352}}, 'edge': {'win_1': {'var': 127.88450704225352, 'med': 137.0, 'ave': 691.0646738962728}, 'win_2': {'var': 128.0465179968701, 'med': 132.0, 'ave': 779.0828517254195}, 'win_3': {'var': 128.27437402190924, 'med': 133.0, 'ave': 722.9041007427367}, 'win_4': {'var': 128.18673708920187, 'med': 130.0, 'ave': 695.5960760513786}, 'win_5': {'var': 127.98861502347418, 'med': 130.0, 'ave': 781.3392678783972}, 'win_6': {'var': 128.10379499217527, 'med': 131.0, 'ave': 649.0503768343411}}, 'hsv': {'win_1': {'var': 184.76494522691706, 'med': 198.0, 'ave': 3424.355782117501}, 'win_2': {'var': 202.4300860719875, 'med': 215.0, 'ave': 2738.5651433415746}, 'win_3': {'var': 216.26795774647888, 'med': 224.0, 'ave': 1404.8300374567436}, 'win_4': {'var': 221.09929577464789, 'med': 228.0, 'ave': 889.2793422270713}, 'win_5': {'var': 224.0598200312989, 'med': 232.0, 'ave': 1200.6441132696457}, 'win_6': {'var': 221.69448356807513, 'med': 226.0, 'ave': 821.0763388960523}}, 'red': {'win_1': {'var': 160.29530516431925, 'med': 169.0, 'ave': 1172.4486321056227}, 'win_2': {'var': 160.17253521126761, 'med': 167.0, 'ave': 770.3223442769294}, 'win_3': {'var': 158.98798904538342, 'med': 162.0, 'ave': 497.0680482252322}, 'win_4': {'var': 158.2757433489828, 'med': 159.0, 'ave': 676.3889108324088}, 'win_5': {'var': 156.89127543035994, 'med': 159.0, 'ave': 804.3305498599752}, 'win_6': {'var': 154.1140453834116, 'med': 156.0, 'ave': 700.7744349650765}}, 'blue': {'win_1': {'var': 138.60661189358373, 'med': 151.0, 'ave': 1598.9243146553204}, 'win_2': {'var': 138.31838810641628, 'med': 148.0, 'ave': 1246.5048887946002}, 'win_3': {'var': 137.14460093896713, 'med': 141.0, 'ave': 588.211250192863}, 'win_4': {'var': 136.2288732394366, 'med': 138.0, 'ave': 923.5304026427739}, 'win_5': {'var': 136.0835289514867, 'med': 139.0, 'ave': 1130.1489305824953}, 'win_6': {'var': 136.02566510172144, 'med': 138.0, 'ave': 854.369137859674}}, 'green': {'win_1': {'var': 198.7435054773083, 'med': 203.0, 'ave': 436.6620665848682}, 'win_2': {'var': 204.23485915492958, 'med': 208.0, 'ave': 475.30200080175894}, 'win_3': {'var': 203.2299295774648, 'med': 205.0, 'ave': 342.60585695904695}, 'win_4': {'var': 198.5005868544601, 'med': 200.0, 'ave': 529.686932362957}, 'win_5': {'var': 199.0907276995305, 'med': 201.0, 'ave': 480.4814789696709}, 'win_6': {'var': 197.28693270735525, 'med': 199.0, 'ave': 378.48793566213834}}, 'purple': {'win_1': {'var': 171.558372456964, 'med': 179.0, 'ave': 919.1760136265341}, 'win_2': {'var': 171.29084507042253, 'med': 178.0, 'ave': 722.9619662655117}, 'win_3': {'var': 168.99256651017214, 'med': 172.0, 'ave': 399.8078477166249}, 'win_4': {'var': 167.31028951486698, 'med': 169.0, 'ave': 572.1623667393424}, 'win_5': {'var': 165.90250391236307, 'med': 168.0, 'ave': 682.2162378618784}, 'win_6': {'var': 163.04941314553992, 'med': 165.0, 'ave': 612.5371123316581}}, 'emerald': {'win_1': {'var': 209.5865805946792, 'med': 212.0, 'ave': 336.5606571652572}, 'win_2': {'var': 214.77402190923317, 'med': 218.0, 'ave': 426.94556927025553}, 'win_3': {'var': 213.20442097026603, 'med': 215.0, 'ave': 302.15754696519406}, 'win_4': {'var': 207.41643192488263, 'med': 209.0, 'ave': 479.81054376336266}, 'win_5': {'var': 207.87077464788732, 'med': 210.0, 'ave': 505.9060316377923}, 'win_6': {'var': 206.15884194053208, 'med': 208.0, 'ave': 365.00505092806884}}, 'yellow': {'win_1': {'var': 223.61306729264476, 'med': 232.0, 'ave': 1808.3141328452368}, 'win_2': {'var': 215.7641236306729, 'med': 237.0, 'ave': 4317.6032434025565}, 'win_3': {'var': 225.32320031298906, 'med': 236.0, 'ave': 2221.4797747345224}, 'win_4': {'var': 218.57676056338028, 'med': 230.0, 'ave': 2471.075484967709}, 'win_5': {'var': 217.74100156494524, 'med': 229.0, 'ave': 2442.710932330201}, 'win_6': {'var': 221.51306729264476, 'med': 226.0, 'ave': 806.2290154743448}}}],
#       dtype=object)
# array([{'normalRGB': {'win_1': {'var': 212.19968701095462, 'med': 239.0, 'ave': 5553.639154831615}, 'win_2': {'var': 173.10872456964006, 'med': 239.0, 'ave': 11324.53727912445}, 'win_3': {'var': 200.63986697965572, 'med': 241.0, 'ave': 7965.048668839896}, 'win_4': {'var': 203.82985133020344, 'med': 235.0, 'ave': 6291.3803999778975}, 'win_5': {'var': 204.32077464788733, 'med': 234.0, 'ave': 6009.342604407745}, 'win_6': {'var': 221.16913145539905, 'med': 233.0, 'ave': 2381.549915677555}}, 'enphasis': {'win_1': {'var': 144.67331768388107, 'med': 140.0, 'ave': 1086.0334977566672}, 'win_2': {'var': 147.10860719874805, 'med': 141.0, 'ave': 2064.551271769025}, 'win_3': {'var': 148.664710485133, 'med': 142.0, 'ave': 1733.9579252291703}, 'win_4': {'var': 146.92766040688576, 'med': 142.0, 'ave': 2137.2495009425793}, 'win_5': {'var': 145.74906103286386, 'med': 141.0, 'ave': 2386.174744814741}, 'win_6': {'var': 147.73595461658843, 'med': 142.0, 'ave': 1636.2184255754053}}, 'edge': {'win_1': {'var': 127.97852112676057, 'med': 136.0, 'ave': 629.6714870148119}, 'win_2': {'var': 128.1399061032864, 'med': 132.0, 'ave': 738.3887204919658}, 'win_3': {'var': 128.135406885759, 'med': 133.0, 'ave': 724.0599513602656}, 'win_4': {'var': 128.2744131455399, 'med': 130.0, 'ave': 693.2387819325971}, 'win_5': {'var': 127.98705007824726, 'med': 131.0, 'ave': 784.1247540522652}, 'win_6': {'var': 128.02249608763694, 'med': 131.0, 'ave': 680.9057928305795}}, 'hsv': {'win_1': {'var': 183.46897496087638, 'med': 197.0, 'ave': 3426.190586742722}, 'win_2': {'var': 200.739593114241, 'med': 214.0, 'ave': 2858.47350280784}, 'win_3': {'var': 216.21471048513303, 'med': 224.0, 'ave': 1408.6427100492015}, 'win_4': {'var': 221.26639280125195, 'med': 227.0, 'ave': 737.9415935608872}, 'win_5': {'var': 224.71944444444443, 'med': 231.0, 'ave': 961.1117815379935}, 'win_6': {'var': 218.70712050078248, 'med': 226.0, 'ave': 1339.6422341185}}, 'red': {'win_1': {'var': 160.13482003129891, 'med': 169.0, 'ave': 1176.2793196467976}, 'win_2': {'var': 159.88974960876368, 'med': 166.0, 'ave': 760.059284600841}, 'win_3': {'var': 158.72922535211268, 'med': 162.0, 'ave': 508.67085166909123}, 'win_4': {'var': 158.03751956181534, 'med': 159.0, 'ave': 691.6537174780993}, 'win_5': {'var': 156.6206572769953, 'med': 159.0, 'ave': 762.4053948731513}, 'win_6': {'var': 153.8733959311424, 'med': 156.0, 'ave': 677.173408029467}}, 'blue': {'win_1': {'var': 138.6012910798122, 'med': 151.0, 'ave': 1595.0499122611254}, 'win_2': {'var': 138.21036776212833, 'med': 148.0, 'ave': 1213.0021069070046}, 'win_3': {'var': 136.98446791862284, 'med': 141.0, 'ave': 600.2224113366077}, 'win_4': {'var': 136.1010172143975, 'med': 138.0, 'ave': 837.3831445051809}, 'win_5': {'var': 136.09385758998434, 'med': 139.0, 'ave': 1135.1482721299542}, 'win_6': {'var': 135.94659624413146, 'med': 139.0, 'ave': 884.1194876319735}}, 'green': {'win_1': {'var': 198.1716744913928, 'med': 202.0, 'ave': 425.82107559983444}, 'win_2': {'var': 203.70974178403756, 'med': 208.0, 'ave': 444.5151633136062}, 'win_3': {'var': 202.59197965571204, 'med': 205.0, 'ave': 335.0032768321369}, 'win_4': {'var': 197.82417840375587, 'med': 199.0, 'ave': 560.4231556238841}, 'win_5': {'var': 198.5413536776213, 'med': 201.0, 'ave': 477.16988611747865}, 'win_6': {'var': 196.7323552425665, 'med': 199.0, 'ave': 404.3596260647261}}, 'purple': {'win_1': {'var': 171.42476525821596, 'med': 179.0, 'ave': 919.1967653987303}, 'win_2': {'var': 171.07480438184663, 'med': 178.0, 'ave': 742.132041237164}, 'win_3': {'var': 168.61334115805946, 'med': 171.0, 'ave': 390.0975606676488}, 'win_4': {'var': 167.00281690140844, 'med': 168.0, 'ave': 560.9068778240648}, 'win_5': {'var': 165.68587636932708, 'med': 168.0, 'ave': 666.2490180504433}, 'win_6': {'var': 162.7435054773083, 'med': 165.0, 'ave': 635.5159007006743}}, 'emerald': {'win_1': {'var': 209.03697183098592, 'med': 212.0, 'ave': 343.0631479506932}, 'win_2': {'var': 214.197965571205, 'med': 217.0, 'ave': 420.1251288814438}, 'win_3': {'var': 212.51420187793428, 'med': 215.0, 'ave': 302.5695166165223}, 'win_4': {'var': 206.7556338028169, 'med': 209.0, 'ave': 513.7477968987635}, 'win_5': {'var': 207.3593896713615, 'med': 210.0, 'ave': 461.1352365602063}, 'win_6': {'var': 205.529186228482, 'med': 207.0, 'ave': 375.8996176476351}}, 'yellow': {'win_1': {'var': 224.3972222222222, 'med': 231.0, 'ave': 1525.0194054294905}, 'win_2': {'var': 217.80586854460094, 'med': 236.0, 'ave': 3802.634300458463}, 'win_3': {'var': 225.90011737089202, 'med': 235.0, 'ave': 1946.8848982647846}, 'win_4': {'var': 217.67136150234742, 'med': 229.0, 'ave': 2481.638945094668}, 'win_5': {'var': 218.12840375586853, 'med': 229.0, 'ave': 2234.748066466089}, 'win_6': {'var': 221.00140845070422, 'med': 225.0, 'ave': 755.6190121007738}}}],
#       dtype=object)
# array([{'normalRGB': {'win_1': {'var': 215.7428012519562, 'med': 231.0, 'ave': 3322.8094356584647}, 'win_2': {'var': 213.8422143974961, 'med': 236.0, 'ave': 4619.194939384393}, 'win_3': {'var': 225.85512519561814, 'med': 234.0, 'ave': 1748.5921959545922}, 'win_4': {'var': 218.22609546165884, 'med': 229.0, 'ave': 2183.8777149580233}, 'win_5': {'var': 217.19248826291079, 'med': 228.0, 'ave': 2273.836422447045}, 'win_6': {'var': 220.1380281690141, 'med': 224.0, 'ave': 818.1440155172035}}, 'enphasis': {'win_1': {'var': 143.520813771518, 'med': 138.0, 'ave': 996.4549658479481}, 'win_2': {'var': 147.67492175273864, 'med': 142.0, 'ave': 1938.0578217857396}, 'win_3': {'var': 148.54084507042253, 'med': 142.0, 'ave': 1569.8861251074522}, 'win_4': {'var': 147.21122848200312, 'med': 142.0, 'ave': 2073.6072995862933}, 'win_5': {'var': 146.2810641627543, 'med': 142.0, 'ave': 2320.178201684459}, 'win_6': {'var': 147.40399061032863, 'med': 141.0, 'ave': 1566.9468072362185}}, 'edge': {'win_1': {'var': 127.8050469483568, 'med': 136.0, 'ave': 607.2761169383278}, 'win_2': {'var': 127.90383411580595, 'med': 133.0, 'ave': 711.2013937502602}, 'win_3': {'var': 128.13834115805946, 'med': 135.0, 'ave': 706.7724892670227}, 'win_4': {'var': 128.09710485133022, 'med': 130.0, 'ave': 669.1473781595853}, 'win_5': {'var': 128.01909233176838, 'med': 131.0, 'ave': 791.8474445595499}, 'win_6': {'var': 128.15050860719876, 'med': 131.0, 'ave': 684.6573549838853}}, 'hsv': {'win_1': {'var': 192.14350547730828, 'med': 199.0, 'ave': 2297.0101573516918}, 'win_2': {'var': 204.81064162754305, 'med': 213.0, 'ave': 2012.3574141423044}, 'win_3': {'var': 213.36267605633802, 'med': 220.0, 'ave': 1252.723004419317}, 'win_4': {'var': 218.04092331768388, 'med': 222.0, 'ave': 548.2660091631338}, 'win_5': {'var': 223.49761345852895, 'med': 229.0, 'ave': 642.580354241822}, 'win_6': {'var': 217.98012519561814, 'med': 222.0, 'ave': 633.8384938810397}}, 'red': {'win_1': {'var': 157.74612676056339, 'med': 167.0, 'ave': 1081.8450554205515}, 'win_2': {'var': 157.2235524256651, 'med': 164.0, 'ave': 746.0371135148571}, 'win_3': {'var': 155.87683881064163, 'med': 159.0, 'ave': 489.78843069545775}, 'win_4': {'var': 155.19542253521126, 'med': 156.0, 'ave': 641.3597364803059}, 'win_5': {'var': 154.00770735524256, 'med': 156.0, 'ave': 780.0248623494138}, 'win_6': {'var': 151.60516431924881, 'med': 154.0, 'ave': 677.4291595582887}}, 'blue': {'win_1': {'var': 137.93763693270736, 'med': 151.0, 'ave': 1562.7528401123136}, 'win_2': {'var': 137.40524256651017, 'med': 147.0, 'ave': 1186.8509584309893}, 'win_3': {'var': 136.21913145539907, 'med': 140.0, 'ave': 537.3959172425003}, 'win_4': {'var': 135.31150234741784, 'med': 137.0, 'ave': 870.1951415614186}, 'win_5': {'var': 135.34659624413146, 'med': 139.0, 'ave': 1009.9133217477794}, 'win_6': {'var': 135.22410015649453, 'med': 138.0, 'ave': 857.4827212168857}}, 'green': {'win_1': {'var': 192.89698748043818, 'med': 197.0, 'ave': 410.73551518137685}, 'win_2': {'var': 197.35086071987482, 'med': 201.0, 'ave': 414.014455440695}, 'win_3': {'var': 195.83779342723005, 'med': 198.0, 'ave': 304.961873691287}, 'win_4': {'var': 191.74127543035993, 'med': 193.0, 'ave': 524.59209915575}, 'win_5': {'var': 192.4046165884194, 'med': 195.0, 'ave': 465.7122635071427}, 'win_6': {'var': 190.28129890453835, 'med': 192.0, 'ave': 406.4966142752883}}, 'purple': {'win_1': {'var': 167.50406885758997, 'med': 175.0, 'ave': 752.4319083270271}, 'win_2': {'var': 167.2949530516432, 'med': 173.0, 'ave': 631.9356552794859}, 'win_3': {'var': 164.93924100156494, 'med': 168.0, 'ave': 377.83672305459424}, 'win_4': {'var': 163.42241784037557, 'med': 165.0, 'ave': 561.3734802260353}, 'win_5': {'var': 162.2294600938967, 'med': 165.0, 'ave': 671.4150710700036}, 'win_6': {'var': 159.72527386541472, 'med': 162.0, 'ave': 583.279298633918}}, 'emerald': {'win_1': {'var': 202.8843896713615, 'med': 205.0, 'ave': 305.19082048039405}, 'win_2': {'var': 207.16697965571205, 'med': 210.0, 'ave': 368.33182045498523}, 'win_3': {'var': 204.7564945226917, 'med': 207.0, 'ave': 287.84289600583855}, 'win_4': {'var': 200.0633020344288, 'med': 202.0, 'ave': 419.8865249338143}, 'win_5': {'var': 200.5940923317684, 'med': 203.0, 'ave': 462.2934158036814}, 'win_6': {'var': 198.50183881064163, 'med': 201.0, 'ave': 353.9466319865375}}, 'yellow': {'win_1': {'var': 221.2003129890454, 'med': 224.0, 'ave': 632.7958684466388}, 'win_2': {'var': 222.25786384976527, 'med': 229.0, 'ave': 1311.1025594431217}, 'win_3': {'var': 223.49589201877933, 'med': 226.0, 'ave': 475.3685277254292}, 'win_4': {'var': 217.69917840375587, 'med': 221.0, 'ave': 788.9292638007229}, 'win_5': {'var': 216.60629890453833, 'med': 221.0, 'ave': 925.9045847369471}, 'win_6': {'var': 214.21381064162753, 'med': 216.0, 'ave': 407.1515854790104}}}],
#       dtype=object)
# array([{'normalRGB': {'win_1': {'var': 215.80892018779343, 'med': 231.0, 'ave': 3227.658089444334}, 'win_2': {'var': 215.3435054773083, 'med': 236.0, 'ave': 4289.546088494101}, 'win_3': {'var': 226.45571205007823, 'med': 234.0, 'ave': 1555.580745932734}, 'win_4': {'var': 219.32359154929577, 'med': 228.0, 'ave': 1835.9651241899755}, 'win_5': {'var': 217.04319248826292, 'med': 228.0, 'ave': 2216.26746148251}, 'win_6': {'var': 220.07104851330203, 'med': 224.0, 'ave': 763.6963605594618}}, 'enphasis': {'win_1': {'var': 143.5702269170579, 'med': 138.0, 'ave': 991.2857567560203}, 'win_2': {'var': 147.65872456964007, 'med': 142.0, 'ave': 1929.0989066674872}, 'win_3': {'var': 148.7147496087637, 'med': 142.0, 'ave': 1570.8966056102306}, 'win_4': {'var': 147.58708920187794, 'med': 143.0, 'ave': 2054.4974232956424}, 'win_5': {'var': 146.4997261345853, 'med': 142.0, 'ave': 2298.72394358697}, 'win_6': {'var': 147.5881455399061, 'med': 142.0, 'ave': 1541.1933258254535}}, 'edge': {'win_1': {'var': 127.82535211267606, 'med': 136.0, 'ave': 624.219654609976}, 'win_2': {'var': 127.8359937402191, 'med': 134.0, 'ave': 835.718641852856}, 'win_3': {'var': 128.04557902973397, 'med': 135.0, 'ave': 782.1291040856948}, 'win_4': {'var': 128.1761345852895, 'med': 130.0, 'ave': 629.1670986735925}, 'win_5': {'var': 127.90414710485133, 'med': 130.0, 'ave': 819.1703896872805}, 'win_6': {'var': 128.05927230046947, 'med': 131.0, 'ave': 702.6635055737398}}, 'hsv': {'win_1': {'var': 192.51384976525821, 'med': 199.0, 'ave': 2259.439949029073}, 'win_2': {'var': 204.8322378716745, 'med': 213.0, 'ave': 2024.8123097024159}, 'win_3': {'var': 213.39389671361502, 'med': 219.0, 'ave': 1240.2210582115542}, 'win_4': {'var': 218.27237871674492, 'med': 223.0, 'ave': 572.7982667986706}, 'win_5': {'var': 223.88787167449138, 'med': 228.0, 'ave': 527.1233427315764}, 'win_6': {'var': 217.525, 'med': 222.0, 'ave': 745.3326300860717}}, 'red': {'win_1': {'var': 157.47773865414712, 'med': 167.0, 'ave': 1092.0148408957045}, 'win_2': {'var': 157.14021909233176, 'med': 164.0, 'ave': 740.9773652102145}, 'win_3': {'var': 155.7851330203443, 'med': 159.0, 'ave': 473.0032844502242}, 'win_4': {'var': 154.81870109546165, 'med': 156.0, 'ave': 661.3922480781052}, 'win_5': {'var': 153.94119718309858, 'med': 156.0, 'ave': 756.972089959554}, 'win_6': {'var': 151.5685054773083, 'med': 154.0, 'ave': 655.2605652155412}}, 'blue': {'win_1': {'var': 137.93603286384976, 'med': 151.0, 'ave': 1585.123255623333}, 'win_2': {'var': 137.2711267605634, 'med': 147.0, 'ave': 1148.270934724151}, 'win_3': {'var': 136.16830985915493, 'med': 140.0, 'ave': 545.4701068460843}, 'win_4': {'var': 135.24107981220658, 'med': 137.0, 'ave': 828.5127725429257}, 'win_5': {'var': 135.29417057902972, 'med': 139.0, 'ave': 1006.519371338665}, 'win_6': {'var': 135.20344287949922, 'med': 138.0, 'ave': 864.7126798523711}}, 'green': {'win_1': {'var': 192.73532863849766, 'med': 197.0, 'ave': 417.87443263847564}, 'win_2': {'var': 197.1105633802817, 'med': 201.0, 'ave': 412.41970062156975}, 'win_3': {'var': 195.50669014084508, 'med': 198.0, 'ave': 301.2397830980405}, 'win_4': {'var': 191.39718309859154, 'med': 193.0, 'ave': 506.5452189821244}, 'win_5': {'var': 192.06784037558685, 'med': 194.0, 'ave': 446.75040550816635}, 'win_6': {'var': 190.11823161189358, 'med': 192.0, 'ave': 392.2775705817238}}, 'purple': {'win_1': {'var': 167.16275430359937, 'med': 175.0, 'ave': 740.3473764513702}, 'win_2': {'var': 167.1778951486698, 'med': 173.0, 'ave': 650.3254346932315}, 'win_3': {'var': 164.77695618153365, 'med': 168.0, 'ave': 372.87908557084984}, 'win_4': {'var': 162.88235524256652, 'med': 164.0, 'ave': 578.7174977392175}, 'win_5': {'var': 162.05023474178404, 'med': 165.0, 'ave': 665.0549882078071}, 'win_6': {'var': 159.6209702660407, 'med': 162.0, 'ave': 589.9482769928561}}, 'emerald': {'win_1': {'var': 202.69718309859155, 'med': 205.0, 'ave': 302.65822367696}, 'win_2': {'var': 206.95097809076682, 'med': 210.0, 'ave': 360.3676672749504}, 'win_3': {'var': 204.45496870109545, 'med': 206.0, 'ave': 286.910648238457}, 'win_4': {'var': 199.5505477308294, 'med': 201.0, 'ave': 450.720293127221}, 'win_5': {'var': 200.28372456964007, 'med': 203.0, 'ave': 428.2097977081757}, 'win_6': {'var': 198.19186228482002, 'med': 200.0, 'ave': 355.90419042860884}}, 'yellow': {'win_1': {'var': 220.67241784037557, 'med': 224.0, 'ave': 636.8496930585862}, 'win_2': {'var': 222.28305946791863, 'med': 228.0, 'ave': 1236.7655346146169}, 'win_3': {'var': 223.26044600938968, 'med': 226.0, 'ave': 458.2252429935639}, 'win_4': {'var': 217.2996870109546, 'med': 221.0, 'ave': 725.7634740804417}, 'win_5': {'var': 216.35743348982786, 'med': 221.0, 'ave': 881.1094087494887}, 'win_6': {'var': 213.92390453834116, 'med': 216.0, 'ave': 397.23094656991674}}}],
#       dtype=object)
# array([{'normalRGB': {'win_1': {'var': 207.65133020344288, 'med': 241.0, 'ave': 6605.640244709431}, 'win_2': {'var': 160.012558685446, 'med': 237.0, 'ave': 12496.942760902268}, 'win_3': {'var': 187.49323161189358, 'med': 241.0, 'ave': 10032.5777319667}, 'win_4': {'var': 197.5043427230047, 'med': 236.0, 'ave': 7546.476585209614}, 'win_5': {'var': 198.05406885758998, 'med': 235.0, 'ave': 7236.703571081331}, 'win_6': {'var': 218.87726917057904, 'med': 234.0, 'ave': 3213.4629105394406}}, 'enphasis': {'win_1': {'var': 144.2908841940532, 'med': 139.0, 'ave': 1101.6757541477782}, 'win_2': {'var': 147.43474178403756, 'med': 142.0, 'ave': 2074.8592781414623}, 'win_3': {'var': 148.71952269170578, 'med': 142.0, 'ave': 1743.0555656563708}, 'win_4': {'var': 147.05258215962442, 'med': 142.0, 'ave': 2155.2214135202453}, 'win_5': {'var': 145.36537558685447, 'med': 141.0, 'ave': 2402.6965085052566}, 'win_6': {'var': 147.75136932707355, 'med': 142.0, 'ave': 1645.9781280153973}}, 'edge': {'win_1': {'var': 127.78517214397496, 'med': 137.0, 'ave': 722.3131995400065}, 'win_2': {'var': 128.06737089201877, 'med': 132.0, 'ave': 780.5956959046928}, 'win_3': {'var': 128.22366979655712, 'med': 133.0, 'ave': 761.3358481914352}, 'win_4': {'var': 128.1721048513302, 'med': 130.0, 'ave': 688.1724534725743}, 'win_5': {'var': 128.12535211267607, 'med': 131.0, 'ave': 709.5678549229648}, 'win_6': {'var': 128.14233176838812, 'med': 131.0, 'ave': 643.2043113077701}}, 'hsv': {'win_1': {'var': 184.15465571205007, 'med': 198.0, 'ave': 3452.9761207343604}, 'win_2': {'var': 201.82320031298906, 'med': 215.0, 'ave': 2801.0239453135523}, 'win_3': {'var': 216.18075117370893, 'med': 224.0, 'ave': 1407.1109127377724}, 'win_4': {'var': 220.43133802816902, 'med': 227.0, 'ave': 1043.647006973374}, 'win_5': {'var': 223.85786384976527, 'med': 232.0, 'ave': 1233.507614216205}, 'win_6': {'var': 221.08196400625977, 'med': 226.0, 'ave': 905.9659344838375}}, 'red': {'win_1': {'var': 160.56244131455398, 'med': 169.0, 'ave': 1169.2299038991382}, 'win_2': {'var': 160.50915492957748, 'med': 167.0, 'ave': 759.1916219775618}, 'win_3': {'var': 159.4277386541471, 'med': 162.0, 'ave': 497.78437141213647}, 'win_4': {'var': 158.73630672926447, 'med': 160.0, 'ave': 759.883908738468}, 'win_5': {'var': 157.29448356807512, 'med': 159.0, 'ave': 784.3727864703872}, 'win_6': {'var': 154.44878716744913, 'med': 156.0, 'ave': 666.2768764633095}}, 'blue': {'win_1': {'var': 138.6975743348983, 'med': 151.0, 'ave': 1579.839602879842}, 'win_2': {'var': 138.4834115805947, 'med': 148.0, 'ave': 1236.7413523673779}, 'win_3': {'var': 137.25762910798122, 'med': 141.0, 'ave': 585.7340576026583}, 'win_4': {'var': 136.34593114241002, 'med': 138.0, 'ave': 843.871098467872}, 'win_5': {'var': 136.20469483568075, 'med': 139.0, 'ave': 1131.0336868787056}, 'win_6': {'var': 136.1586854460094, 'med': 139.0, 'ave': 931.1878862218697}}, 'green': {'win_1': {'var': 199.40485133020346, 'med': 203.0, 'ave': 443.4094130843136}, 'win_2': {'var': 205.00234741784038, 'med': 209.0, 'ave': 409.61298353501286}, 'win_3': {'var': 204.09949139280124, 'med': 206.0, 'ave': 341.9230044361544}, 'win_4': {'var': 199.21361502347418, 'med': 201.0, 'ave': 554.8028819237805}, 'win_5': {'var': 199.9438184663537, 'med': 202.0, 'ave': 481.2642927745572}, 'win_6': {'var': 198.0928012519562, 'med': 200.0, 'ave': 395.3313722781831}}, 'purple': {'win_1': {'var': 171.24413145539907, 'med': 179.0, 'ave': 800.3611353567416}, 'win_2': {'var': 171.737558685446, 'med': 178.0, 'ave': 725.0537380149441}, 'win_3': {'var': 169.53532863849765, 'med': 172.0, 'ave': 397.5369365508387}, 'win_4': {'var': 167.91592331768388, 'med': 169.0, 'ave': 564.2895273556221}, 'win_5': {'var': 166.38212050078246, 'med': 169.0, 'ave': 660.620924454963}, 'win_6': {'var': 163.5025821596244, 'med': 166.0, 'ave': 600.6861435671934}}, 'emerald': {'win_1': {'var': 210.31780125195618, 'med': 213.0, 'ave': 346.9274452437542}, 'win_2': {'var': 215.59659624413146, 'med': 219.0, 'ave': 438.1401996820517}, 'win_3': {'var': 214.258372456964, 'med': 216.0, 'ave': 303.9830871789597}, 'win_4': {'var': 208.20285602503913, 'med': 210.0, 'ave': 514.3196866788016}, 'win_5': {'var': 208.65528169014084, 'med': 211.0, 'ave': 553.9305041851263}, 'win_6': {'var': 207.06349765258216, 'med': 209.0, 'ave': 392.94264253950934}}, 'yellow': {'win_1': {'var': 223.7830985915493, 'med': 233.0, 'ave': 1970.3448943110934}, 'win_2': {'var': 212.98748043818466, 'med': 237.0, 'ave': 5009.250547485924}, 'win_3': {'var': 223.80715962441315, 'med': 237.0, 'ave': 2766.922710868104}, 'win_4': {'var': 217.55363849765257, 'med': 231.0, 'ave': 2900.885464069629}, 'win_5': {'var': 217.07867762128325, 'med': 230.0, 'ave': 2765.5309381574175}, 'win_6': {'var': 222.1507433489828, 'med': 227.0, 'ave': 904.4431997604213}}}],
#       dtype=object)
# array([{'normalRGB': {'win_1': {'var': 203.99444444444444, 'med': 242.0, 'ave': 7410.999264910451}, 'win_2': {'var': 149.90031298904537, 'med': 235.0, 'ave': 13145.506729166513}, 'win_3': {'var': 178.13489827856026, 'med': 241.0, 'ave': 11224.512318886364}, 'win_4': {'var': 191.35508607198747, 'med': 236.0, 'ave': 8560.51108133062}, 'win_5': {'var': 194.21600156494523, 'med': 235.0, 'ave': 7948.326230647884}, 'win_6': {'var': 217.28176838810643, 'med': 235.0, 'ave': 3707.208712991739}}, 'enphasis': {'win_1': {'var': 144.3664710485133, 'med': 140.0, 'ave': 1084.8827959972054}, 'win_2': {'var': 147.3859546165884, 'med': 141.0, 'ave': 2066.5191532749354}, 'win_3': {'var': 148.5098591549296, 'med': 142.0, 'ave': 1733.7481031100529}, 'win_4': {'var': 147.35105633802817, 'med': 143.0, 'ave': 2134.2731209498775}, 'win_5': {'var': 145.25735524256652, 'med': 141.0, 'ave': 2400.3185100631613}, 'win_6': {'var': 147.6482785602504, 'med': 141.0, 'ave': 1641.205869493609}}, 'edge': {'win_1': {'var': 127.88924100156494, 'med': 137.0, 'ave': 729.5923099090544}, 'win_2': {'var': 128.04053208137717, 'med': 132.0, 'ave': 705.528169356952}, 'win_3': {'var': 128.0259780907668, 'med': 132.0, 'ave': 657.0942390668127}, 'win_4': {'var': 128.30958528951487, 'med': 130.0, 'ave': 675.5228189203468}, 'win_5': {'var': 128.0429186228482, 'med': 131.0, 'ave': 779.7873207461163}, 'win_6': {'var': 128.03564162754304, 'med': 130.0, 'ave': 599.781632647782}}, 'hsv': {'win_1': {'var': 188.54237089201877, 'med': 200.0, 'ave': 3062.337328338183}, 'win_2': {'var': 200.90168231611892, 'med': 215.0, 'ave': 2907.459308593912}, 'win_3': {'var': 215.43169014084506, 'med': 223.0, 'ave': 1510.9682602107166}, 'win_4': {'var': 220.5106807511737, 'med': 227.0, 'ave': 1052.4487122126343}, 'win_5': {'var': 224.31126760563382, 'med': 232.0, 'ave': 1129.2839418986532}, 'win_6': {'var': 220.93536776212832, 'med': 226.0, 'ave': 932.6473093717932}}, 'red': {'win_1': {'var': 160.92785602503912, 'med': 169.0, 'ave': 1171.9483946208986}, 'win_2': {'var': 160.9407668231612, 'med': 168.0, 'ave': 784.0560375966459}, 'win_3': {'var': 159.7835289514867, 'med': 163.0, 'ave': 502.01163011301156}, 'win_4': {'var': 159.29381846635368, 'med': 160.0, 'ave': 705.4264250124289}, 'win_5': {'var': 157.7271909233177, 'med': 160.0, 'ave': 789.262625285927}, 'win_6': {'var': 154.81756651017216, 'med': 157.0, 'ave': 752.717226628988}}, 'blue': {'win_1': {'var': 138.81306729264475, 'med': 152.0, 'ave': 1593.3868089015753}, 'win_2': {'var': 138.57519561815337, 'med': 148.0, 'ave': 1219.1052219883375}, 'win_3': {'var': 137.39518779342723, 'med': 141.0, 'ave': 575.3642882667681}, 'win_4': {'var': 136.46075899843507, 'med': 138.0, 'ave': 890.7209953550637}, 'win_5': {'var': 136.3203834115806, 'med': 139.0, 'ave': 1086.3524796652023}, 'win_6': {'var': 136.30176056338027, 'med': 139.0, 'ave': 925.1202472916527}}, 'green': {'win_1': {'var': 200.14053208137716, 'med': 204.0, 'ave': 437.285962784182}, 'win_2': {'var': 205.82824726134587, 'med': 210.0, 'ave': 442.22707376671787}, 'win_3': {'var': 204.9233959311424, 'med': 207.0, 'ave': 335.83591585419316}, 'win_4': {'var': 200.16158059467918, 'med': 202.0, 'ave': 546.4444707411571}, 'win_5': {'var': 200.6488262910798, 'med': 203.0, 'ave': 470.6060197491679}, 'win_6': {'var': 198.83388106416277, 'med': 201.0, 'ave': 390.49204456175414}}, 'purple': {'win_1': {'var': 172.53176838810643, 'med': 180.0, 'ave': 945.3236386568411}, 'win_2': {'var': 172.4092723004695, 'med': 179.0, 'ave': 741.3676683280435}, 'win_3': {'var': 170.12057902973396, 'med': 173.0, 'ave': 403.41370795893437}, 'win_4': {'var': 168.47476525821597, 'med': 170.0, 'ave': 517.3425556960699}, 'win_5': {'var': 166.9804773082942, 'med': 169.0, 'ave': 674.9437111962769}, 'win_6': {'var': 164.0131455399061, 'med': 166.0, 'ave': 590.3992794639512}}, 'emerald': {'win_1': {'var': 210.8654538341158, 'med': 214.0, 'ave': 404.8748941993554}, 'win_2': {'var': 216.50242566510173, 'med': 220.0, 'ave': 450.88066704259643}, 'win_3': {'var': 215.25829420970265, 'med': 217.0, 'ave': 300.5121573406707}, 'win_4': {'var': 209.09835680751175, 'med': 211.0, 'ave': 544.9704511340342}, 'win_5': {'var': 209.7023082942097, 'med': 212.0, 'ave': 521.880119867396}, 'win_6': {'var': 207.95011737089203, 'med': 210.0, 'ave': 383.9319796419361}}, 'yellow': {'win_1': {'var': 223.91205007824726, 'med': 234.0, 'ave': 2161.485926783095}, 'win_2': {'var': 209.7714006259781, 'med': 238.0, 'ave': 5748.147859697089}, 'win_3': {'var': 221.9890062597809, 'med': 238.0, 'ave': 3359.6290262425273}, 'win_4': {'var': 217.5785993740219, 'med': 232.0, 'ave': 3138.89882996313}, 'win_5': {'var': 215.73693270735524, 'med': 231.0, 'ave': 3214.378682923484}, 'win_6': {'var': 222.55395148669797, 'med': 228.0, 'ave': 1030.1739280477245}}}],
#       dtype=object)

# 撮影画像1枚から生成されるtrain_X
# """

# """
# for file in files:
#     pprint(np.load(file,allow_pickle=True)["array_1"][0])
# path=os.getcwd()
# """  # train_X=np.load(path+"/second_input_data/2022-06-09_18-44-40.npz")["array_1"]
# # 特徴画像の数＊特徴量ベクトル＊学習画像の数　を想定
# """
# test_X=train_X[-1].reshape(1,-1)
# train_X=train_X[:-1]
# train_y=np.zeros((train_X.shape[0],1))
# test_y=0
# print(train_X.shape,test_X.shape,train_y.shape)
# train_y[-1:]=1
# model=Lasso(max_iter=1000)
# model.fit(train_X,train_y)
# possibility=model.predict(test_X)
# print(possibility)
# ・Xを1次元にも3次元にもできない問題
# ・今後何らかの支障になるかもなと思い、できればここをロバストにしてみたいと思うところです。
# """
