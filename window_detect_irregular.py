import cv2
import numpy as np
from math import prod
from spmimage.decomposition import KSVD
from PIL import Image
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
import matplotlib.pyplot as plt
import sys

from FEATURE import Feature_img





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

def image_to_Y(img,patch_size,fit=False):
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


def Y_to_image(Y_rec,patch_size,original_img_size):
    """
    入力 : 再構成で生成された画像データ群Y_rec (reconstructed)
          再現したい画像のサイズ(train_img.shape)
    出力 : 再構成画像img_rec
    機能 : image_to_Y()で行われた処理の逆
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


def evaluate(Y,Y_rec_ok,Y_rec_ng,patch_size,original_img_size, img_list, d_num):
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
    plt.imshow(Y_to_image(Y,patch_size,original_img_size))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("train_img (reconstruct)", fontsize=fs)
    plt.subplot(335)
    plt.imshow(Y_to_image(Y_rec_ok,patch_size,original_img_size))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("test_img_ok (reconstruct)", fontsize=fs)
    plt.subplot(336)
    plt.imshow(Y_to_image(Y_rec_ng,patch_size,original_img_size))
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("test_img_ng (reconstruct)", fontsize=fs)
    
    var_org = 0
    var_ok = 0
    var_ng = 0
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            var_org = var_org + ((abs(Y[i][j]-Y[i][j])**2)-(np.average(Y)**2))*1/Y.size
            var_ok = var_ok + ((abs(Y_rec_ok[i][j]-Y[i][j])**2)-(np.average(Y)**2))*1/Y.size
            var_ng = var_ng + ((abs(Y_rec_ng[i][j]-Y[i][j])**2)-(np.average(Y)**2))*1/Y.size
    print(f"元画像分散：{var_org}\nOK画像分散：{var_ok}\nNG画像分散：{var_ng}")
    
    plt.subplot(337)
    plt.hist(abs(Y-Y).reshape(-1,),bins=100,range=(0,10))
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


def img_window(train_img:np.ndarray, test_img_ok:np.ndarray, test_img_ng:np.ndarray, shape:list=(3, 3)):
    """画像探査領域分割関数

    Args:
        train_img (np.ndarray): 画像
        test_img_ok (np.ndarray): 正解テスト用画像
        test_img_ng (np.ndarray): 異常テスト用画像
        shape (list, optional): 所望の領域のシェイプ. Defaults to [3, 3].

    Returns:
        list: 3 lists of separated img.
    """
    train_img_list, test_img_ok_list, test_img_ng_list = [], [], []
    height = train_img.shape[0]
    width = train_img.shape[1]
    
    # 指定の大きさの探査領域を設定
    partial_height = int(height/shape[0])
    partial_width = int(width/shape[1])
    # 探査領域を抽出
    for i in range(shape[0]):
        for j in range(shape[1]):
            train_img_part = train_img[i*partial_height:(i+1)*partial_height, j*partial_width:(j+1)*partial_width]
            test_img_ok_part = test_img_ok[i*partial_height:(i+1)*partial_height, j*partial_width:(j+1)*partial_width]
            test_img_ng_part = test_img_ng[i*partial_height:(i+1)*partial_height, j*partial_width:(j+1)*partial_width]
            
            # まとめて返すためのリストに追加
            train_img_list.append(train_img_part)
            test_img_ok_list.append(test_img_ok_part)
            test_img_ng_list.append(test_img_ng_part)
    
    return train_img_list, test_img_ok_list, test_img_ng_list, (partial_height, partial_width)
    

def estimate(train_img_part, test_img_ok_part, test_img_ng_part, img_size, d_num):
    # 使用画像リスト
    img_list = [train_img_part, test_img_ok_part, test_img_ng_part]
    
    # 学習用画像データ群Yを準備
    Y=image_to_Y(train_img_part,patch_size,fit=True)
    Y_ok=image_to_Y(test_img_ok_part,patch_size,fit=False)
    Y_ng=image_to_Y(test_img_ng_part,patch_size,fit=False)


    # 学習
    D,X,ksvd=generate_dict(Y,n_components,transform_n_nonzero_coefs,max_iter)

    # 推論・画像再構成
    Y_rec_ok=reconstruct_img(Y_ok,D,ksvd)
    Y_rec_ng=reconstruct_img(Y_ng,D,ksvd)

    # 結果表示
    evaluate(Y,Y_rec_ok,Y_rec_ng,patch_size,img_size, img_list, d_num)


def read_img(path_list):
    """画像読込関数
    
    Args:
        path_list (list): 変換希望画像パス一覧
    
    Returns:
        list: 3 paths of featured img.
    """
    global feature_name
    treat = Feature_img(path_list)
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
    train_img = np.asarray(Image.open(path_list[0]).convert('L'))
    test_img_ok=np.asarray(Image.open(path_list[1]).convert('L'))
    test_img_ng=np.asarray(Image.open(path_list[2]).convert('L'))
    train_img = train_img[int(0.5*train_img.shape[0]):]
    test_img_ok = test_img_ok[int(0.5*test_img_ok.shape[0]):]
    test_img_ng = test_img_ng[int(-train_img.shape[0]):, :train_img.shape[1]]
    
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
    return train_img, test_img_ok, test_img_ng    
    

def window_detect(train_img, test_img_ok, test_img_ng):
    # 探査領域の分割数を指定
    detect_shape = (2, 3)

    # 各画像を探査領域に分割してリストに収納
    train_img_list, test_img_ok_list, test_img_ng_list, partial_size = img_window(train_img, test_img_ok, test_img_ng, detect_shape)

    # 各探査領域に対して異常検出を行う
    for k in range(prod(detect_shape)):
        estimate(train_img_list[k], test_img_ok_list[k], test_img_ng_list[k], partial_size, d_num=k+1)


def main():
    # 本編
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
    path_list = ["img_data/data_old/img_test_ok_RPC.jpg",
                "img_data/data_old/img_train_RPC.jpg", 
                "img_data/data_old/img_1.jpg"]
    # edge_Enphasis()
    train_img, test_img_ok, test_img_ng = read_img(path_list)
    window_detect(train_img, test_img_ok, test_img_ng)
    
    
patch_size=(5,5)
n_components=7
transform_n_nonzero_coefs=3
max_iter=15
feature_name = "normal_RGB"


if __name__ == "__main__":
    main()
    