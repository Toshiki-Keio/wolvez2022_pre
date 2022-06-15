import re
import os
import cv2
import numpy as np
from datetime import datetime
from glob import glob
from math import prod

from numpy import save

from baba_into_window import IntoWindow
from bbaa_learn_dict import LearnDict
from bcaa_eval import EvaluateImg

# 一旦一枚目だけ学習
learn_state = True
import_paths = glob("../a_prepare/ac_pictures/aca_normal/movie_1/*")
for path in import_paths:
    now=str(datetime.now())[:19].replace(" ","_").replace(":","-")
    Save = True
    importPath = path.replace("\\", "/")
    saveDir = "b-data"
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    iw_shape = (2, 3)
    #learn_state = True
    D, ksvd = None, None
    dict_list = {}
    feature_values = []


    iw = IntoWindow(importPath, saveDir, Save)

    # processing img
    fmg_list = iw.feature_img(frame_num=now)

    for fmg in fmg_list:
        # breakout by windows
        iw_list, window_size = iw.breakout(iw.read_img(fmg))
        print(fmg)
        feature_name = str(re.findall(saveDir + f"/baca_featuring/(.*)_.*_", fmg)[0])
        print(feature_name, type(feature_name))
        for win in range(int(prod(iw_shape))):
            if learn_state:
                if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                    ld = LearnDict(iw_list[win])
                    D, ksvd = ld.generate()
                    dict_list[feature_name] = [D, ksvd]
                    if not os.path.exists(saveDir + f"/bbba_learnimg"):
                        os.mkdir(saveDir + f"/bbba_learnimg")
                    save_name = saveDir + f"/bbba_learnimg/{feature_name}_part_{win+1}_{now}.jpg"
                    cv2.imwrite(save_name, iw_list[win])
            else:
                print(win)
                D, ksvd = dict_list[feature_name]
                ei = EvaluateImg(iw_list[win])
                img_rec = ei.reconstruct(D, ksvd, window_size)
                ave, med, var = ei.evaluate(iw_list[win], img_rec, win, feature_name, now, saveDir)
                if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                    feature_values.append(ave)
                    feature_values.append(med)
                    feature_values.append(var)
    
    if not os.path.exists(saveDir + f"/bcca_secondinput"):
        os.mkdir(saveDir + f"/bcca_secondinput")
    np.savez_compressed(saveDir + f"/bcca_secondinput/"+now,array_1=np.array(feature_values))
    learn_state = False