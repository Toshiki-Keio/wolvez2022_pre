import numpy as np
#spm-imageはスパースモデリング用のライブラリ。scikit-learnにはまだ同様の機能がないらしい
# cf.: https://codezine.jp/article/detail/11823
from spmimage.decomposition import KSVD
from PIL import Image
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
import matplotlib.pyplot as plt

# スパースモデリングを用いた画像の再構成
# 正常画像を読み込み
ok_img = np.asarray(Image.open("img_data/img_train.jpg").convert('L'))


# 正常画像は再構成できるようにしたいが、それ以外は再構成できないように、表現力を小さく設定する
patch_size = (25,25)
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
print(D.shape,X.shape)
print(X)
reconstructed_patches = np.dot(X, D)
reconstructed_patches = scl.inverse_transform(reconstructed_patches)
reconstructed_patches = reconstructed_patches.reshape(-1, patch_size[0], patch_size[1])
reconstructed_img = reconstruct_from_simple_patches_2d(reconstructed_patches, ok_img.shape)
reconstructed_img[reconstructed_img < 0] = 0
reconstructed_img[reconstructed_img > 255] = 255
reconstructed_img = reconstructed_img.astype(np.uint8)
diff=ok_img-reconstructed_img
diff=abs(diff)
diff=diff.reshape(-1,)
plt.hist(diff)
plt.show()
train_data=reconstructed_img
reconstructed_img=Image.fromarray(reconstructed_img)
reconstructed_img.show()

# 異常画像を読み込み
ng_img = np.asarray(Image.open("img_data/img_test_ng.jpg").convert('L'))

# 異常画像に対するスパースコードを求める
patches = extract_simple_patches_2d(ng_img, patch_size)
patches = patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
print(patches)
Y = scl.transform(patches)
X = ksvd.transform(Y)
reconstructed_patches = np.dot(X, D)
reconstructed_patches = scl.inverse_transform(reconstructed_patches)
reconstructed_patches = reconstructed_patches.reshape(-1, patch_size[0], patch_size[1])
reconstructed_img = reconstruct_from_simple_patches_2d(reconstructed_patches, ok_img.shape)
reconstructed_img[reconstructed_img < 0] = 0
reconstructed_img[reconstructed_img > 255] = 255
reconstructed_img = reconstructed_img.astype(np.uint8)
ng_img_recon=reconstructed_img
reconstructed_img=Image.fromarray(reconstructed_img)
reconstructed_img.show()

diff=ng_img-ng_img_recon
diff=abs(diff)
diff=diff.reshape(-1,)
plt.hist(diff)
plt.show()