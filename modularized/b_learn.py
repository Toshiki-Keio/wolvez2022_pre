import numpy as np
from spmimage.decomposition import KSVD


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