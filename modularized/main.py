from a_read import read_img,img_to_Y
from b_learn import generate_dict
from c_reconstruct import reconstruct_img
import cv2

# 動画から画像を切り出す

# 画像をndarrayに変換する
img_path="wolvez2022/img_data/data_old/img_1.jpg"
img=read_img(img_path)

patch_size=(10,10)

# 画像をpatchに切り分けて、標準化
Y=img_to_Y(img,patch_size)

# 学習
D,X,ksvd=generate_dict(Y,n_components=20,transform_n_nonzero_coefs=3,max_iter=15)
"""
n_components : 生成する基底ベクトルの本数
transform_n_nonzero_coefs : 画像を再構成するために使用を許される基底ベクトルの本数。言い換えれば、Xの非ゼロ成分の個数（L0ノルム）
max_iter : 詳細未詳。学習の反復回数の上限？
"""

img_rec=reconstruct_img(Y,D,ksvd,patch_size,img.shape)

cv2.imshow("original img",img)
cv2.imshow("reconstructed img",img_rec)
cv2.waitKey(0)
cv2.destroyAllWindows()