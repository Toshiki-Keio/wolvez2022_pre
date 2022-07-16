from b__main import spm_first
import os
from glob import glob

alphabet = ord('e')
# ord, chr => a = 97, b = 98, ...


saveDir = "b-data"
if not os.path.exists(saveDir):
    os.mkdir(saveDir)
if not os.path.exists(saveDir + f"/bcca_secondinput"):
    os.mkdir(saveDir + f"/bcca_secondinput")

import_dirs_a = sorted(glob("../a_prepare/ac_pictures/aca_normal/movie_*"))
import_dirs_b = sorted(glob("../a_prepare/ac_pictures/acb_normal_camera_30/movie_*"))
import_dirs_c = sorted(glob("../a_prepare/ac_pictures/acc_sand/movie_*"))
import_dirs_d = sorted(glob("../a_prepare/ac_pictures/acd_slope/movie_*"))
import_dirs_e = sorted(glob("../a_prepare/ac_pictures/ace_wadachi/*_movie_*"))

import_dirs_list = [import_dirs_a, import_dirs_b, import_dirs_c, import_dirs_d, import_dirs_e]

k = 97
for dirs_num, dirs in enumerate(import_dirs_list):
    for mov_num, mov_dir in enumerate(dirs):
        import_paths = mov_dir.replace('\\','/') + '/*.jpg'
        npz_dir = saveDir + f"/bcca_secondinput/bcc{chr(k)}/"
        if not os.path.exists(npz_dir):
            os.mkdir(npz_dir)
        
        spm_first(img_path=import_paths,npz_dir=npz_dir)
        k+=1
