import cv2
import numpy as np
from math import prod
from spmimage.decomposition import KSVD
from PIL import Image
from glob import glob
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
import matplotlib.pyplot as plt
import sys

from FEATURE import Feature_img
from modularized.a_read import read_img, img_to_Y
from modularized.b_learn import generate_dict
from modularized.c_reconstruct import reconstruct_img
from modularized.d_analyze import evaluate





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

'''
def img_to_Y(img,patch_size,fit=False):
    """
    入力 : 取り込んだ画像img
    出力 : 学習・評価に用いる際に用いる画像データ群Y
    機能 : 画像をパッチに分割 -> パッチを2次元から1次元へ変換 -> 1次元ベクトルを標準化

    備考 : 取り込む画像が1枚である必要はないと思う。何枚か（撮影方向を変えるなどして）撮影しておくと、ロバスト性が上がるかもしれません
    注意 : fitは学習の時にTrue、推論の時にFalseとする
    """
    patches=extract_simple_patches_2d(img,patch_size=patch_size)
    patches=patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
    if fit:
        Y=scl.fit_transform(patches)
    else:
        Y=scl.transform(patches)
    return Y
'''

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

'''
def generate_dict(Y,n_components,transform_n_nonzero_coefs,max_iter):
    """
    入力 : 学習画像データ群Y
    出力 : 辞書D・スパースコード（=抽出行列α）X・学習したモデルksvd
    機能 : 画像データ群Yを構成する基底ベクトルの集合Dを生成
          Yを再構成するためにどうDの中身を組み合わせるか、を示すXも生成
          この際、基底ベクトルの数をn_componentsで指定できる
          また、各画素の再構成に使える基底ベクトルの本数をtransform_n_nonzero_coefsで指定できる
          max_iterは詳細不明
    """
    # 学習モデルの定義
    ksvd = KSVD(n_components=n_components, transform_n_nonzero_coefs=transform_n_nonzero_coefs, max_iter=max_iter)
    # 抽出行列を求める
    X = ksvd.fit_transform(Y)
    # 辞書を求める
    D = ksvd.components_
        
    return D,X,ksvd
'''

'''
def reconstruct_img(Y,D,ksvd):
    """
    入力 : 画像のデータ群Y
          学習済みの辞書D
          学習したモデルksvd
    出力 : Dを用いて再構成された画像のデータ群Y_rec (reconstructed)
    機能 : Y~=DXとなるようなXを求める -> Y_rec=DXを求める
    """
    X=ksvd.transform(Y)
    Y_rec=np.dot(X,D)
    return Y_rec
'''

'''
def evaluate(Y_rec_img,Y_rec_ok_img,Y_rec_ng_img,patch_size,original_img_size, img_list, d_num):
    global feature_name
    """
    学習画像・正常画像・異常画像それぞれについて、
    ・元画像
    ・再構成画像
    ・画素値の偏差のヒストグラム
    を出力
    """
    pxcels = prod(original_img_size)
    fs = 10
    plt.subplot(331)
    plt.imshow(img_list[0])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("train_img (original)", fontsize=fs)
    plt.subplot(332)
    plt.imshow(img_list[1])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("test_img_ok (original)", fontsize=fs)
    plt.subplot(333)
    plt.imshow(img_list[2])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("test_img_ng (original)", fontsize=fs)
    plt.subplot(334)
    plt.imshow(Y_rec_img)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("train_img (reconstruct)", fontsize=fs)
    plt.subplot(335)
    plt.imshow(Y_rec_ok_img)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("test_img_ok (reconstruct)", fontsize=fs)
    plt.subplot(336)
    plt.imshow(Y_rec_ng_img)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("test_img_ng (reconstruct)", fontsize=fs)
    
    var_org = 0
    var_ok = 0
    var_ng = 0
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            var_org = var_org + ((abs(Y[i][j]-Y_rec_img[i][j])**2)-(np.average(Y_rec_img)**2))*1/Y.size
            var_ok = var_ok + ((abs(Y_rec_ok_img[i][j]-Y[i][j])**2)-(np.average(Y_rec_ok_img)**2))*1/Y.size
            var_ng = var_ng + ((abs(Y_rec_ng_img[i][j]-Y[i][j])**2)-(np.average(Y_rec_ng_img)**2))*1/Y.size
    print(f"元画像分散：{var_org}\nOK画像分散：{var_ok}\nNG画像分散：{var_ng}\n")
    
    plt.subplot(337)
    plt.hist(abs(-Y).reshape(-1,),bins=100,range=(0,10))
    plt.ylim(0,pxcels/3)
    plt.title("difference", fontsize=fs)
    plt.subplot(338)
    plt.hist(abs(Y_rec_ok-Y).reshape(-1,),bins=100,range=(0,10))
    #print(Y)
    plt.ylim(0,pxcels/3)
    plt.title("difference", fontsize=fs)
    plt.subplot(339)
    plt.hist(abs(Y_rec_ng-Y).reshape(-1,),bins=100,range=(0,10))
    plt.ylim(0,pxcels/3)
    plt.title("difference", fontsize=fs)
    plt.savefig(f"results_data/{feature_name}_part_{d_num}")
    
    #plt.show()
    plt.close()
    #print(np.average(abs(Y_rec_ok-Y)).reshape(-1,)) # 評価方法要検討
    #print(np.average(abs(Y_rec_ng-Y)).reshape(-1,))
'''

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

def estimate(train_img_part:np.ndarray):
    global patch_size
    # 学習用画像データ群Yを準備
    Y=img_to_Y(train_img_part,patch_size)

    # 学習
    D,X,ksvd=generate_dict(Y,n_components=20,transform_n_nonzero_coefs=3,max_iter=15)
    
    return D, ksvd

    

def guess_img(test_img,D,ksvd,patch_size,img_size, d_num):
    Y_test=img_to_Y(test_img,patch_size)
    # 推論・画像再構成
    #Y_rec_img=reconstruct_img(Y,D,ksvd,patch_size,img_size)
    Y_rec_test_img=reconstruct_img(Y_test,D,ksvd,patch_size,img_size)
    # 結果表示
    ave, med, var = evaluate(test_img, Y_rec_test_img, d_num)
    return ave, med, var

def feature_img(path_list, frame_num):
    """画像読込関数
    
    Args:
        path_list (list): 変換希望画像パス一覧
    
    Returns:
        list: 3 paths of featured img.
    """
    global feature_name
    treat = Feature_img(path_list, frame_num)
    feature = sys.argv
    if len(feature)<2:
        pass
    
    elif feature[1]=="vari":
        feature_name = feature[1]
        treat.vari()
        path_list = treat.output()
    
    elif feature[1]=="enphasis":
        feature_name = feature[1]
        treat.enphasis()
        path_list = treat.output()
        
    elif feature[1]=="edge":
        feature_name = feature[1]
        treat.edge()
        path_list = treat.output()
        
    elif feature[1]=="r":
        feature_name = feature[1]
        treat.r()
        path_list = treat.output()
        
    elif feature[1]=="b":
        feature_name = feature[1]
        treat.b()
        path_list = treat.output()
        
    elif feature[1]=="g":
        feature_name = feature[1]
        treat.g()
        path_list = treat.output()
        
    elif feature[1]=="rb":
        feature_name = feature[1]
        treat.rb()
        path_list = treat.output()
        
    elif feature[1]=="gb":
        feature_name = feature[1]
        treat.gb()
        path_list = treat.output()
        
    elif feature[1]=="rg":
        feature_name = feature[1]
        treat.rg()
        path_list = treat.output()
        
    else:
        print(f"{feature[1]} was not found in Feature_img Function.\nFeatures are vari, enphasis, edge, red, blue, green, or nothing as normal.")
        sys.exit()
        
    # 一旦二分の一で画像上部排除
    img = read_img(path_list)
    #test_img_ok=read_img(path_list[1])
    #test_img_ng=read_img(path_list[2])
    img = img[int(0.5*img.shape[0]):]
    #test_img_ok = test_img_ok[int(0.5*test_img_ok.shape[0]):]
    #test_img_ng = test_img_ng[int(-train_img.shape[0]):, :train_img.shape[1]]
    
    # 画像を導入
    """
    edge_mode=False
    if edge_mode:
        train_img = np.asarray(Image.open("img_data/tochigi4_edge.jpg").convert('L'))
        test_img_ok=np.asarray(Image.open("img_data/tochigi5_edge.jpg").convert('L'))
        test_img_ng=np.asarray(Image.open("img_data/tochigi7_edge.jpg").convert('L'))
    else:
        # 一旦二分の一で画像上部排除
        train_img = np.asarray(Image.open("img_data/use_img/img_train_RPC.jpg").convert('L'))
        train_img = train_img[int(0.5*train_img.shape[0]):]
        test_img_ok=np.asarray(Image.open("img_data/data_old/img_test_ok_RPC.jpg").convert('L'))
        test_img_ok = test_img_ok[int(0.5*test_img_ok.shape[0]):]
        test_img_ng=np.asarray(Image.open("img_data/data_old/img_1.jpg").convert('L'))
        test_img_ng = test_img_ng[int(-train_img.shape[0]):, :train_img.shape[1]]
    """
    return img    
    

def detect_window(img):
    # 探査領域の分割数を指定
    detect_shape = (2, 3)

    # 各画像を探査領域に分割してリストに収納
    img_window_list = img_window(img, detect_shape)

    return img_window_list
    # 各探査領域に対して異常検出を行う
    for k in range(prod(detect_shape)):
        D, ksvd = estimate(train_img_list[k], partial_size, d_num=k+1)


def main(img_path, state, times):
    # 本編
    global D, ksvd, feature_name, var_log, ave_log, med_log
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
    img = feature_img(img_path, times)
    
    detect_shape = (2, 3)

    # 各画像を探査領域に分割してリストに収納
    img_window_list, partial_size = img_window(img, detect_shape)
    
    # 各探査領域に対して異常検出を行う
    for k in range(prod(detect_shape)):
        if state:
            if k == 4:
                D, ksvd = estimate(img_window_list[k])
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
        
    
 
feature_name = "normal_RGB"
   
D, ksvd = None, None
var_log, frames = [], []
ave_log = []
med_log = []
patch_size=(5,5)
n_components=7
transform_n_nonzero_coefs=3
max_iter=15


times = 0
if __name__ == "__main__":
    state = True
    pathlist = glob("img_data/from_mov/*")
    while times <= 203:
        main(pathlist[times].replace("\\","/"), state, times)
        frames.append(times)
        times = times + 2
        state = False

frames = np.array(frames)



def logger(data_title:str, log_data):
    global frames
    log_data = np.array(log_data)
    max_data = np.max(log_data)
    pos = np.where(log_data == max_data)
    max_data_frame = int(frames[pos])
    plt.subplot(311)
    plt.plot(frames, log_data, color='g', label="LOG")
    plt.scatter(max_data_frame, max_data, marker="o", s=600, c="yellow", alpha=0.5, edgecolors="r", label=f"Highest {data_title}")
    plt.xlabel("Frame Number")
    plt.ylabel(f"{data_title}")
    plt.xlim(0, int(np.max(frames)))
    plt.ylim(0, max_data+1000)
    plt.title(f"Log of {data_title}\non the Way Passed Through")
    plt.grid(True)
    plt.subplot(313)
    plt.imshow(cv2.imread(f'img_data/from_mov/frame_{max_data_frame}.jpg', 1))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(f"Photo of the above point\nFrame Number is {max_data_frame}")
    plt.savefig(f"results_data/LOG_{data_title}/{feature_name}_{data_title}_log")
    plt.close()
    

logger("Variance", var_log)
logger("Average", ave_log)
logger("Median", med_log)