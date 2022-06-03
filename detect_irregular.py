import numpy as np
from spmimage.decomposition import KSVD
from PIL import Image
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
import matplotlib.pyplot as plt

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

https://codezine.jp/article/detail/11823
https://codezine.jp/article/detail/12433
"""

scl=StandardScaler()

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

def reconstruct_img(Y,D,ksvd,trans=False):
    """
    入力 : 画像のデータ群Y
          学習済みの辞書D
          学習したモデルksvd
    出力 : Dを用いて再構成された画像のデータ群Y_rec (reconstructed)
    機能 : Y~=DXとなるようなXを求める -> Y_rec=DXを求める
    """
    if trans:
        X=ksvd.transform(Y)
    Y_rec=np.dot(X,D)
    return Y_rec

def evaluate(Y,Y_rec_ok,Y_rec_ng,patch_size,original_img_size):
    """
    学習画像・正常画像・異常画像それぞれについて、
    ・元画像
    ・再構成画像
    ・画素値の偏差のヒストグラム
    を出力
    """
    global train_img,test_img_ok,test_img_ng
    plt.subplot(331)
    plt.imshow(train_img)
    plt.title("train_img (original)")
    plt.subplot(332)
    plt.imshow(test_img_ok)
    plt.title("test_img_ok (original)")
    plt.subplot(333)
    plt.imshow(test_img_ng)
    plt.title("test_img_ng (original)")
    plt.subplot(334)
    plt.imshow(Y_to_image(Y,patch_size,original_img_size))
    plt.title("train_img (reconstruct)")
    plt.subplot(335)
    plt.imshow(Y_to_image(Y_rec_ok,patch_size,original_img_size))
    plt.title("test_img_ng (reconstruct)")
    plt.subplot(336)
    plt.imshow(Y_to_image(Y_rec_ng,patch_size,original_img_size))
    plt.title("test_img_ng (reconstruct)")
    plt.subplot(337)
    plt.hist(abs(Y-Y).reshape(-1,),bins=100,range=(0,10))
    plt.ylim(0,50000)
    plt.title("difference")
    plt.subplot(338)
    plt.hist(abs(Y_rec_ok-Y).reshape(-1,),bins=100,range=(0,10))
    plt.ylim(0,50000)
    plt.title("difference")
    plt.subplot(339)
    plt.hist(abs(Y_rec_ng-Y).reshape(-1,),bins=100,range=(0,10))
    plt.ylim(0,50000)
    plt.title("difference")
    plt.show()
    print(np.average(abs(Y_rec_ok-Y)).reshape(-1,)) # 評価方法要検討
    print(np.average(abs(Y_rec_ng-Y)).reshape(-1,))

"""
（学習に関するパラメータについて）
patch_size : 学習用の画像を学習のために分割した際の、分割された画像(=patch)１つ１つのサイズ
n_components : 生成する基底ベクトルの本数
transform_n_nonzero_coefs : 画像を再構成するために使用を許される基底ベクトルの本数。言い換えれば、Xの非ゼロ成分の個数（L0ノルム）
max_iter : 詳細未詳。学習の反復回数の上限？
"""
patch_size=(15,15)
n_components=5
transform_n_nonzero_coefs=3
max_iter=15

"""
（用いる画像について）
train_img   : 学習に用いる画像（１枚のみ）。スタック「しない」状況の画像
test_img_ok : スタック「しない」状況のためのテスト用画像
test_img_ng : スタック「する」状況(=異常)のためのテスト用画像

＊全てモノクロに直して処理。
"""
# 画像を導入

"""
### 変更履歴 ###
"""
edge_mode=False
if edge_mode:
    train_img = np.asarray(Image.open("img_data/img_train_edge.jpg").convert('L'))
    test_img_ok=np.asarray(Image.open("img_data/img_test_ok_edge.jpg").convert('L'))
    test_img_ng=np.asarray(Image.open("img_data/img_test_ng_edge.jpg").convert('L'))
else:
    train_img = np.asarray(Image.open("img_data/img_train.jpg").convert('L'))
    test_img_ok=np.asarray(Image.open("img_data/img_test_ok.jpg").convert('L'))
    test_img_ng=np.asarray(Image.open("img_data/img_test_ng.jpg").convert('L'))
train_img = np.asarray(Image.open("img_data/img_train_edge.jpg").convert('L'))
test_img_ok=np.asarray(Image.open("img_data/img_test_ok_edge.jpg").convert('L'))
test_img_ng=np.asarray(Image.open("img_data/img_test_ng_edge.jpg").convert('L'))
print(type(train_img))

# 学習用画像データ群Yを準備
Y=img_to_Y(train_img,patch_size,fit=True)
Y_ok=img_to_Y(test_img_ok,patch_size,fit=False)
Y_ng=img_to_Y(test_img_ng,patch_size,fit=False)

############################################ デバッグここまでやりました ############################################

# 学習
D,X,ksvd=generate_dict(Y,n_components,transform_n_nonzero_coefs,max_iter)

# 推論・画像再構成
Y_rec_ok=reconstruct_img(Y_ok,D,ksvd,trans=True)
Y_rec_ng=reconstruct_img(Y_ng,D,ksvd,trans=True)

# 結果表示
evaluate(Y,Y_rec_ok,Y_rec_ng,patch_size,train_img.shape)