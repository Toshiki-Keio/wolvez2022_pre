import re
import os
import cv2
import numpy as np
import pickle
from datetime import datetime
from glob import glob
from math import prod

from numpy import save
from regex import P

from baba_into_window import IntoWindow
from bbaa_learn_dict import LearnDict
from bcaa_eval import EvaluateImg

from time import time


# 一旦一枚目だけ学習
learn_state = True
import_paths = glob("../a_prepare/ac_pictures/aca_normal/movie_1/*.jpg")
#import_paths = import_paths[:10]
dict_list = {}
saveDir = "b-data"

if not os.path.exists(saveDir):
    os.mkdir(saveDir)
if not os.path.exists(saveDir + f"/bbba_learnimg"):
    os.mkdir(saveDir + f"/bbba_learnimg")
if not os.path.exists(saveDir + f"/bcca_secondinput"):
    os.mkdir(saveDir + f"/bcca_secondinput")

for path in range(len(import_paths)):
    start_time = time()
    
    now=str(datetime.now())[:19].replace(" ","_").replace(":","-")
    Save = True
    
    # Path that img will be read
    #importPath = path.replace("\\", "/")
    importPath = f"../a_prepare/ac_pictures/aca_normal/movie_1/frame_{path}.jpg".replace("\\","/")
    
    # This will change such as datetime
    print("CURRENT FRAME: "+str(re.findall(".*/frame_(.*).jpg", importPath)[0]))
    
    iw_shape = (2, 3)
    D, ksvd = None, None
    feature_values = {}


    iw = IntoWindow(importPath, saveDir, Save)

    # processing img
    fmg_list = iw.feature_img(frame_num=now)

    for fmg in fmg_list:
        # breakout by windows
        iw_list, window_size = iw.breakout(iw.read_img(fmg))
        feature_name = str(re.findall(saveDir + f"/baca_featuring/(.*)_.*_", fmg)[0])
        print("FEATURED BY: ",feature_name)
        if learn_state:
            print("Learning Phase")
        else:
            print("Evaluate Phase")
        for win in range(int(prod(iw_shape))):
            print("PRAT: ",win+1)
            if learn_state:
                if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                    ld = LearnDict(iw_list[win])
                    D, ksvd = ld.generate()
                    dict_list[feature_name] = [D, ksvd]
                    save_name = saveDir + f"/bbba_learnimg/{feature_name}_part_{win+1}_{now}.jpg"
                    cv2.imwrite(save_name, iw_list[win])
            else:
                D, ksvd = dict_list[feature_name]
                ei = EvaluateImg(iw_list[win])
                img_rec = ei.reconstruct(D, ksvd, window_size)
                saveName = saveDir + f"/bcba_difference"
                if not os.path.exists(saveName):
                    os.mkdir(saveName)
                saveName = saveDir + f"/bcba_difference/{now}"
                if not os.path.exists(saveName):
                    os.mkdir(saveName)
                ave, med, var = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, saveDir)
                if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                    feature_values[feature_name] = {}
                    feature_values[feature_name]["var"] = ave
                    feature_values[feature_name]["med"] = med
                    feature_values[feature_name]["ave"] = var
                '''
                feature_values[feature_name] = {}
                feature_values[feature_name][f'win_{win+1}'] = {}
                feature_values[feature_name]["var"] = ave
                feature_values[feature_name]["med"] = med
                feature_values[feature_name]["ave"] = var
                '''
    if not learn_state:
        np.savez_compressed(saveDir + f"/bcca_secondinput/"+now,array_1=np.array([feature_values]))
        #with open(saveDir + f"/bcca_secondinput/"+now, "wb") as tf:
        #    pickle.dump(feature_values, tf)
    
    end_time = time()
    # Learn state should be changed by main.py
    learn_state = False
    frame = str(re.findall(".*/frame_(.*).jpg", importPath)[0])
    print(f"\n\n==={now}_data was evaluated===\nframe number is {frame}.\nIt cost {end_time-start_time} seconds.\n\n")