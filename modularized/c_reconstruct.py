import numpy as np
from spmimage.feature_extraction.image import reconstruct_from_simple_patches_2d
from sklearn.preprocessing import StandardScaler


def reconstruct_img(Y,D,ksvd,patch_size,original_img_size):
    print("===== func reconstruct_img starts =====")
    X=ksvd.transform(Y)
    Y_rec=np.dot(X,D)
    print("Y was reconstructed by D")
    scl=StandardScaler()
    scl.fit(Y_rec) # おまじない
    # 0-255の画素値に戻す
    Y_rec=scl.inverse_transform(Y_rec)*255/(Y_rec.max()-Y_rec.min())+255/2
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