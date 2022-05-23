from a_read import read_img,img_to_Y

# 動画から画像を切り出す

# 画像をndarrayに変換する
img_path="img_data/data_old/img_1.jpg"
img=read_img(img_path)

# 画像をpatchに切り分けて、標準化
Y=img_to_Y(img,patch_size=(10,10))





