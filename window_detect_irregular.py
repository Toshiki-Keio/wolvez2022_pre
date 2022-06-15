import cv2
import os
import numpy as np
from math import prod
from spmimage.decomposition import KSVD
from PIL import Image
from glob import glob
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
import matplotlib.pyplot as plt
import sys
from datetime import datetime

from FEATURE import Feature_img
from modularized.a_read import read_img, img_to_Y
from modularized.b_learn import generate_dict
from modularized.c_reconstruct import reconstruct_img




"""
（学習の原理について）
与えられたモノクロ画像（正常な時の画像）を幾つにも分割してpatchを作る。
学習に使えるよう、patchに標準化等を施したものがY。
patch全てに共通する基底ベクトルを求め、それを２次元配列でまとめたのがD行列。これを辞書という。
辞書に書かれた基底ベクトルから画像を生成するとき、どのベクトルを使うかを表すのが抽出行列X。(前回のMTGではαと呼ばれていたもの)

学習は正常な時の画像に対して行われているから、導出されたDは「正常な時の画像を生成するのに最適な辞書」である。
だから、異常なデータに対してこのDを使うと、正しく画像が再構成されなくなる。
正しく再構成されたか否かを見極めることで、画像が正常か異常かを見極めることができる。

正しく再構成できたか否かを見極める方法は以下の通り。
元画像の配列と再構成後の画像の配列の差分diffをとり、そのdiffの成分の値をヒストグラムにしてみる。
ヒストグラムが左に寄っていれば、元画像と再構成後の画像がよく似ていることを意味する。つまり正常。
逆に、ヒストグラムが右に寄っていれば、元画像と再構成後の画像が似ていないことを意味する。つまり異常。
"""

scl=StandardScaler()

def Y_to_image(Y_rec,patch_size,original_img_size):
    """
    入力 : 再構成で生成された画像データ群Y_rec (reconstructed)
          再現したい画像のサイズ(train_img.shape)
    出力 : 再構成画像img_rec
    機能 : img_to_Y()で行われた処理の逆
    """
    # 標準化処理の解除
    Y_rec=scl.inverse_transform(Y_rec)
    # 配列の整形
    Y_rec=Y_rec.reshape(-1,patch_size[0],patch_size[1])
    # 画像の復元
    img_rec=reconstruct_from_simple_patches_2d(Y_rec,original_img_size)
    # エラーの修正
    img_rec[img_rec<0]=0
    img_rec[img_rec>255]=255
    # 型の指定
    img_rec=img_rec.astype(np.uint8)
    return img_rec

# 探査領域分割関数
def img_window(img:np.ndarray, shape:list=(3, 3)):
    """画像探査領域分割関数

    Args:
        train_img (np.ndarray): 画像
        test_img_ok (np.ndarray): 正解テスト用画像
        test_img_ng (np.ndarray): 異常テスト用画像
        shape (list, optional): 所望の領域のシェイプ. Defaults to [3, 3].

    Returns:
        list: 3 lists of separated img.
    """
    img_window_list = []
    height = img.shape[0]
    width = img.shape[1]
    
    # 指定の大きさの探査領域を設定
    partial_height = int(height/shape[0])
    partial_width = int(width/shape[1])
    # 探査領域を抽出
    for i in range(shape[0]):
        for j in range(shape[1]):
            img_part = img[i*partial_height:(i+1)*partial_height, j*partial_width:(j+1)*partial_width]
            
            # まとめて返すためのリストに追加
            img_window_list.append(img_part)
    
    return img_window_list, (partial_height, partial_width)

# 評価用ヒストグラム作成
def evaluate(img,img_rec,d_num):
    global times
    """
    学習画像・正常画像・異常画像それぞれについて、
    ・元画像
    ・再構成画像
    ・画素値の偏差のヒストグラム
    を出力
    """
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))
    ax1.imshow(img, cmap='gray')
    ax1.set_title("original img")
    ax2.imshow(img_rec, cmap='gray')
    ax2.set_title("reconstructed img")
    diff=abs(img-img_rec)
    ax3.imshow(diff*255,cmap='gray')
    ax3.set_title("difference")
    ax4.hist(diff.reshape(-1,),bins=255,range=(0,255))
    ax4.set_title("histgram")
    #save_title=str(datetime.datetime.now()).replace(" ","_").replace(":","-")
    #plt.savefig(os.getcwd()+"/img_result/"+save_title+".png")
    plt.savefig(f"results_data/{feature_name}/{feature_name}_{times}thFRAME_part_{d_num}")
    print("average: ",np.average(diff))
    print("median: ",np.median(diff))
    print("variance: ",np.var(diff))
    return np.average(diff),np.median(diff),np.var(diff)   

# 学習
def estimate(train_img_part:np.ndarray):
    global patch_size
    # 学習用画像データ群Yを準備
    Y=img_to_Y(train_img_part,patch_size)

    # 学習
    D,X,ksvd=generate_dict(Y,n_components=20,transform_n_nonzero_coefs=3,max_iter=15)
    
    return D, ksvd

# 辞書を使用した再構成及び構成誤差ヒストグラム作成
def guess_img(test_img,D,ksvd,patch_size,img_size, d_num):
    Y_test=img_to_Y(test_img,patch_size)
    # 推論・画像再構成
    #Y_rec_img=reconstruct_img(Y,D,ksvd,patch_size,img_size)
    Y_rec_test_img=reconstruct_img(Y_test,D,ksvd,patch_size,img_size)
    # 結果表示
    ave, med, var = evaluate(test_img, Y_rec_test_img, d_num)
    return ave, med, var

# 画像を特徴抽出済み画像へ変換
# 標準入力で特徴量抽出可能
# いずれはなくなる関数
def feature_img(path_list, frame_num, feature_name):
    """画像読込関数
    
    Args:
        path_list (str): 変換希望画像パス一覧
    
    Returns:
        str: path of featured img.
    """
    # 「パスリスト」と書いてしまったがこれは一つのパス
    treat = Feature_img(path_list, frame_num)
    if feature_name=="vari":
        treat.vari()
        path_list = treat.output()
    
    elif feature_name=="enphasis":
        treat.enphasis()
        path_list = treat.output()
        
    elif feature_name=="edge":
        treat.edge()
        path_list = treat.output()
        
    elif feature_name=="r":
        treat.r()
        path_list = treat.output()
        
    elif feature_name=="b":
        treat.b()
        path_list = treat.output()
        
    elif feature_name=="g":
        treat.g()
        path_list = treat.output()
        
    elif feature_name=="rb":
        treat.rb()
        path_list = treat.output()
        
    elif feature_name=="gb":
        treat.gb()
        path_list = treat.output()
        
    elif feature_name=="rg":
        treat.rg()
        path_list = treat.output()
        
    else:
        pass
        
    # 一旦二分の一で画像上部排除
    img = read_img(path_list)
    img = img[int(0.5*img.shape[0]):]
    
    return img    

# 経路ログ出力関数
# 最も危険だと思われる箇所を表示(仮)
def logger(data_title:str, log_data):
    global frames
    log_data = np.array(log_data)
    max_data = np.max(log_data)
    pos = np.where(log_data == max_data)
    max_data_frame = int(frames[pos][0])
    plt.subplot(311)
    plt.plot(frames, log_data, color='g', label="LOG")
    plt.scatter(max_data_frame, max_data, marker="o", s=400, c="yellow", alpha=0.5, edgecolors="r", label=f"Highest {data_title}")
    plt.xlabel("Frame Number")
    plt.ylabel(f"{data_title}")
    plt.xlim(0, int(np.max(frames)))
    plt.ylim(0, max_data+1000)
    plt.title(f"Log of {data_title}\non the Way Passed Through")
    plt.grid(True)
    plt.legend()
    plt.subplot(313)
    plt.imshow(cv2.imread(f'img_data/from_mov/frame_{max_data_frame}.jpg', 1))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(f"Photo of above point\nFrame Number is {max_data_frame}")
    plt.savefig(f"results_data/LOG_{data_title}/{feature_name}_{data_title}_log")
    plt.close()

    """
    2段階目に出力されるデータのログ採取
    """
    number_log = f"number_log/LOG_{data_title}"
    if os.path.exists(number_log):
        pass
    else:
        os.mkdir(number_log)
    np.savez_compressed(number_log+f"/{feature_name}_{data_title}_log",array_1=log_data,array_2=max_data)


def make_npz(frame_points):
    frame_points = np.array(frame_points)
    now=str(datetime.now())[:19].replace(" ","_").replace(":","-")
    np.savez_compressed("second_input_data/"+now,array_1=frame_points)

# 特徴抽出から評価ヒストグラム作成まで
def main(img_path, state, times, feature_name):
    # 本編
    global D, ksvd, var_log, ave_log, med_log
    """
    （学習に関するパラメータについて）
    patch_size : 学習用の画像を学習のために分割した際の、分割された画像(=patch)１つ１つのサイズ
    n_components : 生成する基底ベクトルの本数
    transform_n_nonzero_coefs : 画像を再構成するために使用を許される基底ベクトルの本数。言い換えれば、Xの非ゼロ成分の個数（L0ノルム）
    max_iter : 詳細未詳。学習の反復回数の上限？
    """
    """
    （用いる画像について）
    train_img   : 学習に用いる画像（１枚のみ）。スタック「しない」状況の画像
    test_img_ok : スタック「しない」状況のためのテスト用画像
    test_img_ng : スタック「する」状況(=異常)のためのテスト用画像

    ＊全てモノクロに直して処理。
    """
    #path_list = ["img_data/data_old/img_test_ok_RPC.jpg",
    #            "img_data/data_old/img_train_RPC.jpg", 
    #            "img_data/data_old/img_1.jpg"]
    # edge_Enphasis()
    img = feature_img(img_path, times, feature_name)
    
    detect_shape = (2, 3)

    # 各画像を探査領域に分割してリストに収納
    img_window_list, partial_size = img_window(img, detect_shape)
    
    # 各探査領域に対して異常検出を行う
    for k in range(prod(detect_shape)):
        if state:
            if k == 4:
                D, ksvd = estimate(img_window_list[k])
                print(type(D), type(ksvd))
                dict_list[feature_name] = [D, ksvd]
                save_name = f"img_data/use_img/learn_img/{feature_name}_part_{k}.jpg"
                cv2.imwrite(save_name, img_window_list[k])
        try:
            ave, med, var = guess_img(img_window_list[k],D,ksvd,patch_size,partial_size, d_num=k+1)
            if k == 4:
                var_log.append(var)
                ave_log.append(ave)
                med_log.append(med)
        except:
            continue
    return var, ave, med
        
    
# 特徴抽出デフォルト
feature_name = "normal_RGB"


# ログ用リスト
var_log, frames = [], []
ave_log = []
med_log = []


# 辞書学習用パラメータ
patch_size=(5,5)
n_components=7
transform_n_nonzero_coefs=3
max_iter=15
D, ksvd = None, None
dict_list = {}

# フレームごとに異常検出
times = 0  # ループ番号(フレーム番号)
if __name__ == "__main__":
    state = True
    pathlist = glob("img_data/from_mov/*")
    
    # 一旦全フレーム回すループ
    while times <= len(pathlist):
        frame_points = []
        for feature_name in ["normal_RGB", "vari", "enphasis", "edge", "r", "g", "b", "rg", "rb", "gb"]:
        # 一旦最初だけ学習ステートへ
            try:
                var, ave, med = main(f"img_data/from_mov/frame_{times}.jpg", state, times, feature_name)
            except:
                # for ubuntu (temporary)
                var, ave, med = main(f"img_data/from_mov/frame_{times}.jpg", state, times, feature_name)
            frames.append(times)
            print(f"\n\n{times}th frame has done\n\n")
            
            frame_points.append(var)
            frame_points.append(ave)
            frame_points.append(med)
            # 画像が変わらないフレームがいくつか存在
            # 一旦2フレームに一回
        times = times + 100
        state = False

        make_npz(frame_points)
        #frames = np.array(frames)

        
        # ログ出力
        #logger("Variance", var_log)
        #print("\n\nSaved Variance Log\n\n")
        #logger("Average", ave_log)
        #print("\n\nSaved Average Log\n\n")
        #logger("Median", med_log)
        #print("\n\nSaved Median Log\n\n")