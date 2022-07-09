import glob
from cbaa_second_spm import SPM2Open_npz, SPM2Learn, SPM2Evaluate

npz_dir="hogehoge/hogehoge"

train_npz=sorted(glob.glob(npz_dir+"/*"))

spm2_prepare=SPM2Open_npz()
data_list_all_win,label_list_all_win=spm2_prepare.unpack(train_npz)

spm2_learn=SPM2Learn()

#ウィンドウによってスタックと教示する時間帯を変えず、一括とする場合
stack_start=11
stack_end=11

# ウィンドウによってスタックと教示する時間帯を時間帯を変える場合はnp.arrayを定義
stack_info=None
"""
    stack_info=np.array([[12., 18.],
        [12., 18.],
        [12., 18.],
        [12., 18.],
        [12., 18.],
        [12, 18.]])
    「stackした」と学習させるフレームの指定方法
    1. 全ウィンドウで一斉にラベリングする場合
        Learnの引数でstack_startおよびstack_endを[s]で指定する。
    2. ウィンドウごとに個別にラベリングする場合
    stack_info=np.array(
        [
            [win_1_stack_start,win_1_stack_end],
            [win_2_stack_start,win_2_stack_end],
            ...
            [win_6_stack_start,win_6_stack_end],
        ]
    )
    t[s]で入力すること。
"""
spm2_learn.start(data_list_all_win,label_list_all_win,fps=30,stack_appear=stack_start,stack_disappear=stack_end,stack_info=stack_info)

model_master,label_list_all_win,scaler_master=spm2_learn.get_data()
"""
model_master: 各ウィンドウを学習したモデル（俗にいう'model.predict()'とかの'modelに相当するのがリストで入ってる）
label_list_all_win: 重み行列の各成分を、その意味（ex: window_1のrgb画像のaverage)の説明で置き換えた配列
scaler_master: 各ウィンドウを標準化をしたときのモデル（scaler.transform()の'scaler'に相当するのがリストで入ってる）
"""


spm2_predict_prepare=SPM2Open_npz()
test_data_list_all_win,test_label_list_all_win=spm2_predict_prepare.unpack()

spm2_predict=SPM2Evaluate()
spm2_predict.start(model_master,test_data_list_all_win,test_label_list_all_win,scaler_master,train_code,test_code)
score=spm2_predict.get_score()