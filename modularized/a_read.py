"""
画像の下処理関連
"""
import cv2
import numpy as np
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d
from sklearn.preprocessing import StandardScaler

def read_img(path):
    print("===== func read_img starts =====")
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # 読み込めないエラーが生じた際のロバスト性も検討したい
    return img

def img_to_Y(img,patch_size=(10,10)):
    scl=StandardScaler()
    print("===== func img_to_Y starts =====")
    print("shape of img: ",img.shape)
    patches=extract_simple_patches_2d(img,patch_size=patch_size)# 画像をpatch_sizeに分割
    patches=patches.reshape(-1, np.prod(patch_size))# 2次元に直す。(枚数,patchの積) つまりパッチを2→1次元にしている
    print("patch_size: ",patches.shape)# (枚数,patch_size[0],patch_size[1])つまり３じげん
    Y=scl.fit_transform(patches)# 各パッチの標準化（スケールの違いを標準化する）
    print("patches were standardized")
    return Y

"""
scl=StandardScaler()について

scl.fit(data)下準備？
scl.transform(data)で標準化。fit_transformでfitも同時にできる
scl.inverse_transform(data)でもとに戻す
2次元以下の配列にのみ対応。
"""

