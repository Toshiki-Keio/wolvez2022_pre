import numpy as np
#spm-imageはスパースモデリング用のライブラリ。scikit-learnにはまだ同様の機能がないらしい
# cf.: https://codezine.jp/article/detail/11823
from spmimage.decomposition import KSVD
from PIL import Image
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d


# スパースモデリングを用いた画像の再構成

# 画像の読み込み
print("# 画像の読み込み")
img = np.asarray(Image.open("img_data/img_train.jpg").convert('L'))

# 画像はパッチに細分化され、観測値として学習に使用される。まずパッチサイズを指定
print("# 画像はパッチに細分化され、観測値として学習に使用される。まずパッチサイズを指定")
patch_size = (5, 5)
# パッチの切り出し
print("# パッチの切り出し")
patches = extract_simple_patches_2d(img, patch_size)
# パッチのベクトル化
print("# パッチのベクトル化")
patches = patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
# 過学習防止に必須な標準化をパッチに施し、パッチの集合をYとする
print("# 過学習防止に必須な標準化をパッチに施し、パッチの集合をYとする")
scl = StandardScaler()
Y = scl.fit_transform(patches)
"""
ここから、スパースモデリングの要の辞書学習。
辞書学習とは、いくつか集めてきた観測値Yに対し、Y=DXなるDとXを求めること。
DはYを構成する'要素'の集合。これを辞書という。
Xは、Dの中から本質的なものを選び出すための行列。（前回はalphaとしていたものに相当。）
"""
print("ksvdスタート")
ksvd = KSVD(n_components=50, transform_n_nonzero_coefs=5)
print("calculate X")
X = ksvd.fit_transform(Y)
print("calculate D")
D = ksvd.components_
