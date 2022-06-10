from pprint import pprint
import numpy as np
import pandas as pd
import glob
import os
import datetime

path=os.getcwd()
dirs=glob.glob(path+"/number_log/**/*")
all_data={}# 特徴量ごと
for file in dirs:
    file_name=os.path.basename(file)
    all_data[file_name]=[np.load(file)["array_1"],np.load(file)["array_2"]]

keys=all_data.keys()
pic_data=[]# 画像ごとに整理
for i in range(len(all_data[file_name][0])):
    pic=[]
    for key in keys:
        pic.append(all_data[key][0][i])# その画像の特徴量（平均値とか中央値とか）
    pic_data.append(pic)
pic_data=np.array(pic_data)
print(pic_data.shape)

now=str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")
np.savez_compressed(path+"/second_input_data/"+now,array_1=pic_data)