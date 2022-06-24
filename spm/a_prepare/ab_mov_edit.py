import os
import cv2
import traceback
root_path = 'spm/a_prepare/aa_movies/'#おおもとのパス
folder_list = os.listdir(path=root_path)
folder_list = [filename for filename in folder_list if not filename.startswith('.')]#Delete .DS_store
output_path = "spm/a_prepare/ac_pictures/"

for folder in folder_list:
    os.makedirs(output_path+str(folder),exist_ok=False)#出力フォルダ作成。
    folder_path = root_path+str(folder)
    movie_list = os.listdir(path=folder_path)
    movie_list = [filename for filename in movie_list if not filename.startswith('.')]#Delete .DS_store
    print("movie_list:",movie_list)
    mov_count = 0
    for movie in movie_list:
        movie_path = folder_path+"/"+str(movie)
        os.makedirs(output_path+str(folder)+"/movie_"+str(mov_count),exist_ok=False)#make output_folder
        print("movie_path",movie_path)
        cap=cv2.VideoCapture(movie_path)
        print("cap:",cap)
        
        n = 0
        while True:
            ret, frame = cap.read()

            if ret:
                print("spm/a_prepare/ac_pictures/{}/movie_{}/frame_{}.jpg".format(folder,mov_count,n))
                print(frame.shape[0],frame.shape[1])
                cv2.imwrite("spm/a_prepare/ac_pictures/{}/movie_{}/frame_{}.jpg".format(folder,mov_count,n),frame)
                n += 1
            else: 
                break
        mov_count += 1