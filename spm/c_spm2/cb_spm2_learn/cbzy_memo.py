import os
all_path="/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/b_spm1/b-data/bczz_h_param/psize_('005', '005')-ncom_001-tcoef_001-mxiter_001.npz"
filename=os.path.basename(all_path)
print(filename)
patch=int(filename[8:11])
ncom=int(filename[26:29])
tcoef=int(filename[36:39])
print(tcoef)