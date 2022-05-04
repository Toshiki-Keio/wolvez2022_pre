import numpy as np
#spm-imageはスパースモデリング用のライブラリ。scikit-learnにはまだ同様の機能がないらしい
# cf.: https://codezine.jp/article/detail/11823
from spmimage.decomposition import KSVD
from PIL import Image
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d


# スパースモデリングを用いた画像の再構成
# 正常画像を読み込み
ok_img = np.asarray(Image.open("img_data/img_train.jpg").convert('L'))


# 正常画像は再構成できるようにしたいが、それ以外は再構成できないように、表現力を小さく設定する
patch_size = (5, 5)
n_components = 10
transform_n_nonzero_coefs = 3
max_iter=15

# 学習用データの用意
scl = StandardScaler()
patches = extract_simple_patches_2d(ok_img, patch_size)
patches = patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
Y = scl.fit_transform(patches)

# 辞書学習
ksvd = KSVD(n_components=n_components, transform_n_nonzero_coefs=transform_n_nonzero_coefs, max_iter=max_iter)
X = ksvd.fit_transform(Y)
D = ksvd.components_