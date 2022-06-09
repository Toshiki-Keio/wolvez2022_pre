from pprint import pprint
import numpy as np
import glob
import os

dirs=glob.glob("wolvez2022/number_log/**/*")
all_data={}
for file in dirs:
    file_name=os.path.basename(file)
    all_data[file_name]=[np.load(file)["array_1"],np.load(file)["array_2"]]
pprint(all_data)

