# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os
import glob
from pprint import pprint

class Open_npz():
    def __init__(self,files):
        pass
    
    pass



# wolvez2022/spmで実行してください
spm_path = os.getcwd()
train_files = sorted(glob.glob(spm_path+"/b_spm1/b-data/bcca_secondinput/*"))
train_X=Open_npz(train_files)
