import numpy as np
from spmimage.decomposition import KSVD
from PIL import Image
from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
import matplotlib.pyplot as plt

"""
用いる画像について
train_img   : 学習に用いる画像（１枚のみ）。スタック「しない」状況の画像
test_img_ok : スタック「しない」状況のためのテスト用画像
test_img_ng : スタック「する」状況(=異常)のためのテスト用画像
"""
train_img = np.asarray(Image.open("img_data/img_train.jpg").convert('L'))
test_img_ok=np.asarray(Image.open("img_data/img_test_ok.jpg").convert('L'))
test_img_ng=np.asarray(Image.open("img_data/img_test_ng.jpg").convert('L'))


# 学習
"""
学習について
train_imgから、patch_sizeの大きさをもつベクトルpatchesを作る
patchesは、train_imgを構成する基底ベクトル群。
patchesは最初2次元画像がたくさん集まった集合だが、これを一次元化してベクトルにしている。
patchesは正規化して、
"""
patch_size=(25,25)
patches=extract_simple_patches_2d(train_img,patch_size=patch_size)
patches=patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
