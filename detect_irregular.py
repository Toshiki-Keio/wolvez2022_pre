import numpy as np
from spmimage.decomposition import KSVD
from PIL import Image
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
import matplotlib.pyplot as plt

"""
（用いる画像について）
train_img   : 学習に用いる画像（１枚のみ）。スタック「しない」状況の画像
test_img_ok : スタック「しない」状況のためのテスト用画像
test_img_ng : スタック「する」状況(=異常)のためのテスト用画像

＊全てモノクロに直して処理。
"""
train_img = np.asarray(Image.open("img_data/img_train.jpg").convert('L'))
test_img_ok=np.asarray(Image.open("img_data/img_test_ok.jpg").convert('L'))
test_img_ng=np.asarray(Image.open("img_data/img_test_ng.jpg").convert('L'))

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

"""
学習に関するパラメータについて
patch_size : 学習用の画像を学習のために分割した際の、分割された画像(=patch)１つ１つのサイズ
n_components : 生成する基底ベクトルの本数
transform_n_nonzero_coefs : 画像を再構成するために使用を許される基底ベクトルの本数。言い換えれば、Xの非ゼロ成分の個数（L0ノルム）
max_iter : 詳細未詳。学習の反復回数の上限？
"""
patch_size=(25,25)
n_components=100
transform_n_nonzero_coefs=10
max_iter=15
# 学習
"""
学習材料
1枚(or数枚?)のtrain_img(正常時の画像)を分割して、patchesを作る
patchesは最初2次元画像がたくさん集まった集合だが、これを一次元化・正規化してベクトルにしている。
"""

def image_to_Y(img,patch_size,fit=False):
    """
    入力 : 取り込んだ画像img
    出力 : 学習・評価に用いる際に用いる画像データ群Y
    機能 : 画像をパッチに分割 -> パッチを2次元から1次元へ変換 -> 1次元ベクトルを標準化

    備考 : 取り込む画像が1枚である必要はないと思う。何枚か（撮影方向を変えるなどして）撮影しておくと、ロバスト性が上がるかもしれません
    注意 : fitは学習の時にTrue、推論の時にFalseとする
    """
    patches=extract_simple_patches_2d(train_img,patch_size=patch_size)
    patches=patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
    if fit:
        Y=StandardScaler().fit_transform(patches)
    else:
        Y=StandardScaler().transform(patches)
    return Y

def Y_to_image(Y_rec,patch_size,original_img_size):
    """
    入力 : 再構成で生成された画像データ群Y_rec (reconstructed)
          再現したい画像のサイズ(train_img.shape)
    出力 : 再構成画像img_rec
    機能 : image_to_Y()で行われた処理の逆
    """
    # 標準化処理の解除
    Y_rec=StandardScaler().inverse_transform(Y_rec)
    # 配列の整形
    Y_rec=Y_rec.reshape(-1,patch_size[0],patch_size[1])
    # 画像の復元
    img_rec=reconstruct_from_simple_patches_2d(Y_rec,original_img_size)
    # エラーの修正
    img_rec[img_rec<0]=0
    img_rec[img_rec>255]=255
    # 型の指定
    img_rec=img_rec.astype(np.uint8)
    # 可視化
    img_rec=Image.fromarray(img_rec)
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

def evaluate(Y,Y_rec):
    
    pass

# 正常画像(訓練データとは別物)を上記で作ったDを用いて再構成してみる
# スパースコードX_okを求める
patches = extract_simple_patches_2d(test_img_ok, patch_size)
patches = patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
Y_ok= scl.transform(patches)
X_ok = ksvd.transform(Y_ok)# Xだけ作る

# 直前で作ったX_okと、あらかじめ学習で作ったDを用いて画像を再構成する

reconstructed_patches = np.dot(X_ok, D)
reconstructed_patches = scl.inverse_transform(reconstructed_patches)
reconstructed_patches = reconstructed_patches.reshape(-1, patch_size[0], patch_size[1])
reconstructed_img = reconstruct_from_simple_patches_2d(reconstructed_patches, test_img_ok.shape)
reconstructed_img[reconstructed_img < 0] = 0
reconstructed_img[reconstructed_img > 255] = 255
reconstructed_img = reconstructed_img.astype(np.uint8)
diff=test_img_ok-reconstructed_img
diff=abs(diff)
print(diff)
print(diff.shape)
diff=diff.reshape(-1,)
plt.hist(diff)
plt.show()
train_data=reconstructed_img
reconstructed_img=Image.fromarray(reconstructed_img)
reconstructed_img.show()

# 正常画像(訓練データとは別物)を上記で作ったDを用いて再構成してみる
# スパースコードX_okを求める
patches = extract_simple_patches_2d(test_img_ng, patch_size)
patches = patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
Y_ng= scl.transform(patches)
X_ng = ksvd.transform(Y_ng)# Xだけ作る

# 直前で作ったX_ngと、あらかじめ学習で作ったDを用いて画像を再構成する
print(D)
print(X_ng)
reconstructed_patches = np.dot(X_ng, D)
reconstructed_patches = scl.inverse_transform(reconstructed_patches)
reconstructed_patches = reconstructed_patches.reshape(-1, patch_size[0], patch_size[1])
reconstructed_img = reconstruct_from_simple_patches_2d(reconstructed_patches, test_img_ng.shape)
reconstructed_img[reconstructed_img < 0] = 0
reconstructed_img[reconstructed_img > 255] = 255
reconstructed_img = reconstructed_img.astype(np.uint8)
diff=test_img_ng-reconstructed_img
diff=abs(diff)
print(diff)
print(diff.shape)
diff=diff.reshape(-1,)
plt.hist(diff)
plt.show()
train_data=reconstructed_img
reconstructed_img=Image.fromarray(reconstructed_img)
reconstructed_img.show()