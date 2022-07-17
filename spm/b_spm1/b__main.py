import re
import os
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from datetime import datetime
from glob import glob
from math import prod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from baba_into_window import IntoWindow
from bbaa_learn_dict import LearnDict
from bcaa_eval import EvaluateImg

from time import time


'''
直列処理をする場合に呼ばれるプログラム
'''


def spm_first(img_path=False,npz_dir=None, learn_state=False,patch_size=(24,32),n_components=5,transform_n_nonzero_coefs=4,max_iter=15):
    # 一旦一枚目だけ学習
    learn_state = True
    # import_paths = sorted(glob("../a_prepare/ac_pictures/aca_normal/movie_3/*.jpg"))
    if not img_path:
        import_paths = sorted(glob('test/*.jpg'))
    else:
        import_paths = sorted(glob(img_path))
    if len(import_paths) > 1000:
        import_paths = import_paths[:1000]
    import_paths = import_paths# ここの[:10]を外しましたby林出
    dict_list = {}
    saveDir = "b-data"

    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    if not os.path.exists(saveDir + f"/bbba_learnimg"):
        os.mkdir(saveDir + f"/bbba_learnimg")
    if not os.path.exists(saveDir + f"/bcca_secondinput"):
        os.mkdir(saveDir + f"/bcca_secondinput")
    saveName = saveDir + f"/bcba_difference"
    if not os.path.exists(saveName):
        os.mkdir(saveName)

    for path in import_paths:
        start_time = time()
        
        now=str(datetime.now())[:19].replace(" ","_").replace(":","-")
        print(now)
        saveName = saveDir + f"/bcba_difference/{now}"
        if not os.path.exists(saveName):
            os.mkdir(saveName)
        Save = True
        
        # Path that img will be read
        importPath = path.replace("\\", "/")
        
        # This will change such as datetime
        print("CURRENT FRAME: "+str(re.findall(".*/frame_(.*).jpg", importPath)[0]))
        
        iw_shape = (2, 3)
        D, ksvd = None, None
        feature_values = {}

        if learn_state:
            print("=====LEARNING PHASE=====")
        else:
            print("=====EVALUATING PHASE=====")
            
        temp_dir = TemporaryDirectory()
        temp_dir_name = temp_dir.name.replace('//', '/').replace("\\","/")
        iw = IntoWindow(importPath, temp_dir_name, Save)
        # processing img
        fmg_list = iw.feature_img(frame_num=now)
        
        for fmg in fmg_list:
            # breakout by windows
            iw_list, window_size = iw.breakout(iw.read_img(fmg))
            feature_name = str(re.findall(temp_dir_name + f"/baca_featuring/(.*)_.*_", fmg)[0])
            print("FEATURED BY: ",feature_name)
            for win in range(int(prod(iw_shape))):
                #print("PRAT: ",win+1)
                if learn_state:
                    if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                        ld = LearnDict(iw_list[win],patch_size=patch_size,n_components=n_components,transform_n_nonzero_coefs=transform_n_nonzero_coefs,max_iter=max_iter)
                        D, ksvd = ld.generate()
                        dict_list[feature_name] = [D, ksvd]
                        save_name = saveDir + f"/bbba_learnimg/{feature_name}_part_{win+1}_{now}.jpg"
                        cv2.imwrite(save_name, iw_list[win])
                        
                        params = f"psize_{(str(ld.patch_size[0]).zfill(3),str(ld.patch_size[1]).zfill(3))}-ncom_{str(ld.n_components).zfill(3)}-tcoef_{str(ld.transform_n_nonzero_coefs).zfill(3)}-mxiter_{str(ld.max_iter).zfill(3)}"
                else:
                    D, ksvd = dict_list[feature_name]
                    ei = EvaluateImg(iw_list[win],patch_size=patch_size,n_components=n_components,transform_n_nonzero_coefs=transform_n_nonzero_coefs,max_iter=max_iter)
                    img_rec = ei.reconstruct(D, ksvd, window_size)
                    saveName = saveDir + f"/bcba_difference"
                    if not os.path.exists(saveName):
                        os.mkdir(saveName)
                    saveName = saveDir + f"/bcba_difference/{now}"
                    if not os.path.exists(saveName):
                        os.mkdir(saveName)
                    ave, med, var, mode, kurt, skew = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, saveDir)
                    #if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                    #    feature_values[feature_name] = {}
                    #    feature_values[feature_name]["var"] = ave
                    #    feature_values[feature_name]["med"] = med
                    #    feature_values[feature_name]["ave"] = var
                    
                    if  win == 0:
                        feature_values[feature_name] = {}
                    feature_values[feature_name][f'win_{win+1}'] = {}
                    feature_values[feature_name][f'win_{win+1}']["var"] = ave
                    feature_values[feature_name][f'win_{win+1}']["med"] = med
                    feature_values[feature_name][f'win_{win+1}']["ave"] = var
                    feature_values[feature_name][f'win_{win+1}']["mode"] = mode
                    feature_values[feature_name][f'win_{win+1}']["kurt"] = kurt  # 尖度
                    feature_values[feature_name][f'win_{win+1}']["skew"] = skew  # 歪度
        
        
                    
        if not learn_state:
            np.savez_compressed(npz_dir+now,array_1=np.array([feature_values]))
            # np.savez_compressed(saveDir + f"/bczz_h_param/{params}",array_1=np.array([feature_values]))
            #with open(saveDir + f"/bcca_secondinput/"+now, "wb") as tf:
            #    pickle.dump(feature_values, tf)
        
        end_time = time()
        # Learn state should be changed by main.py
        learn_state = False
        frame = str(re.findall(".*/frame_(.*).jpg", importPath)[0])
        print(f"\n\n==={now}_data was evaluated===\nframe number is {frame}.\nIt cost {end_time-start_time} seconds.\n\n")
        temp_dir.cleanup()

patch=60
n_components=5
transform_n_nonzero_coefs=4
max_iter=15

if __name__ == "__main__":
    #    spm_first(patch_size=(patch,patch),n_components=n_components,transform_n_nonzero_coefs=transform_n_nonzero_coefs,max_iter=max_iter)
    #    spm_first(patch_size=(patch,patch),n_components=n_components,transform_n_nonzero_coefs=transform_n_nonzero_coefs,max_iter=max_iter)

    # for patch in range(5,105,5):
    #     for n_components in range(1,patch+1,2):
    #         for transform_n_nonzero_coefs in range(1,n_components+1,2):
    spm_first(patch_size=(40,71),n_components=n_components,transform_n_nonzero_coefs=transform_n_nonzero_coefs,max_iter=max_iter)
    #             spm_first(patch_size=(40,71),n_components=n_components,transform_n_nonzero_coefs=transform_n_nonzero_coefs,max_iter=max_iter)
