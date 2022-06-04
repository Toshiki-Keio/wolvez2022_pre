import numpy as np
import cv2

from PIL import Image
from matplotlib import pyplot as plt
from time import time

class Feature_img():
    output_img_list = None
    save_name = None
    def __init__(self, path_list, frame_num):
        self.path_list = path_list
        self.frame_num = frame_num

    # 赤色抽出
    def r(self):
        #self.output_img_list = []
        self.org_img = np.asarray(Image.open(self.path_list))
        self.org_img[:, :, 1] = 0
        self.org_img[:, :, 2] = 0
        self.save_name = f"img_data/use_img/red_{self.frame_num}thFRAME.jpg"
        self.output_img = Image.fromarray(self.org_img)
        self.output_img.save(self.save_name)
        #cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)

    # 青色抽出
    def b(self):
        #self.output_img_list = []
        self.org_img = np.asarray(Image.open(self.path_list))
        self.org_img[:, :, 0] = 0
        self.org_img[:, :, 1] = 0
        self.save_name = f"img_data/use_img/blue_{self.frame_num}thFRAME.jpg"
        self.output_img = Image.fromarray(self.org_img)
        self.output_img.save(self.save_name)
        #cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)

    # 緑色抽出
    def g(self):
        #self.output_img_list = []
        self.org_img = np.asarray(Image.open(self.path_list))
        self.org_img[:, :, 0] = 0
        self.org_img[:, :, 2] = 0
        self.save_name = f"img_data/use_img/green_{self.frame_num}thFRAME.jpg"
        self.output_img = Image.fromarray(self.org_img)
        self.output_img.save(self.save_name)
        #cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)

    # 緑色排除
    def rb(self):
        #self.output_img_list = []
        self.org_img = np.asarray(Image.open(self.path_list))
        self.org_img[:, :, 1] = 0
        self.save_name = f"img_data/use_img/purple_{self.frame_num}thFRAME.jpg"
        self.output_img = Image.fromarray(self.org_img)
        self.output_img.save(self.save_name)
        #cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)

    # 赤色排除
    def gb(self):
        #self.output_img_list = []
        self.org_img = np.asarray(Image.open(self.path_list))
        self.org_img[:, :, 0] = 0
        self.save_name = f"img_data/use_img/emerald_{self.frame_num}thFRAME.jpg"
        self.output_img = Image.fromarray(self.org_img)
        self.output_img.save(self.save_name)
        #cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)
    
    # 青色排除
    def rg(self):
        #self.output_img_list = []
        self.org_img = np.asarray(Image.open(self.path_list))
        self.org_img[:, :, 2] = 0
        self.save_name = f"img_data/use_img/yellow_{self.frame_num}thFRAME.jpg"
        self.output_img = Image.fromarray(self.org_img)
        self.output_img.save(self.save_name)
        #cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)

    # VARI
    def vari(self):
        #self.output_img_list = []
        self.org_img = cv2.imread(self.path_list,1)
        self.vari_list_np = np.ones((self.org_img.shape[0],self.org_img.shape[1]), np.float64)
        self.output_img = np.ones((self.org_img.shape[0],self.org_img.shape[1]), np.uint8)
        for i in range(self.org_img.shape[0]):
            for j in range(self.org_img.shape[1]):
                vari = 0.0
                b = float(self.org_img[i][j][0])
                g = float(self.org_img[i][j][1])
                r = float(self.org_img[i][j][2])
                if b < 125:
                    vari_d = g+r-b
                    if vari_d != 0:
                        vari = (g-r)/(g+r-b)
                        if vari < 0.0:
                            vari = 0.0
                else:
                    vari = 0
                # vari = vari*255/9.0
                self.vari_list_np[i][j] = vari

        vari_max = np.amax(self.vari_list_np)
        vari_min = np.amin(self.vari_list_np)
        #print("vari max: "+str(np.amax(self.vari_list_np)))
        #print("vari min: "+str(np.amin(self.vari_list_np)))
        for i in range(self.org_img.shape[0]):
            for j in range(self.org_img.shape[1]):
                self.vari_list_np[i][j] = 100*(self.vari_list_np[i][j] - vari_min)/(vari_max - vari_min)
                if self.vari_list_np[i][j] > 1.0:
                    self.vari_list_np[i][j] = 1.0
                self.vari_list_np[i][j] = 255*self.vari_list_np[i][j]
                # print(self.vari_list_np[i][j])
                # print(np.uint8(self.vari_list_np[i][j]))
                self.output_img[i][j] = np.uint8(self.vari_list_np[i][j])
        #print("len(self.vari_list_np): "+str(self.vari_list_np.shape))
        #print("vari max: "+str(np.amax(self.vari_list_np)))
        #print("vari min: "+str(np.amin(self.vari_list_np)))
        #print(self.vari_list_np)

        #cv2.imshow("self.org_img", cv2.resize(self.org_img,dsize=(534,400)))
        #cv2.imshow("VARI_img",cv2.resize(self.output_img,dsize=(534,460)))

        self.save_name = f"img_data/use_img/vari_{self.frame_num}thFRAME.jpg"
        cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)
    
    # エッジ強調
    def enphasis(self):
        #self.output_img_list = []
        self.org_img = cv2.imread(self.path_list, 1)
        self.org_img = cv2.cvtColor(self.org_img, cv2.COLOR_BGR2RGB)
        kernel = np.array([[0, 2, 0],
                            [2, -8, 2],
                            [0, 2, 0]], np.float32)
        self.output_img = cv2.filter2D(self.org_img, -1, kernel)
        self.save_name = f"img_data/use_img/edge_enphasis_{self.frame_num}thFRAME.jpg"
        cv2.imwrite(self.save_name, self.output_img)
        #self.output_img_list.append(self.save_name)
    
    # エッジ検出
    def edge(self):
        #self.output_img_list = []
        self.org_img = cv2.imread(self.path_list)
        self.img_gray = cv2.cvtColor(self.org_img, cv2.COLOR_BGR2GRAY)
        self.gray=cv2.Canny(self.img_gray,100,200)
        self.save_name = f"img_data/use_img/edge_{self.frame_num}thFRAME.jpg"
        cv2.imwrite(self.save_name,self.gray)
        #self.output_img_list.append(self.save_name)
    
    # 特徴抽出済画像パス引き渡し
    def output(self):
        return self.save_name
    
    # 作ってみたが使わない
    def show(self):
        k = 0
        for img in self.output_img_list:
            plt.subplot(int(f"1{len(self.output_img_list)}{k}"))
            plt.imshow(img)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.title(f"img_{self.frame_num}thFRAME")
            k += 1
        plt.show()




if __name__ == "__main__":
    feat = Feature_img(["img_data/data_old/img_train_RPC.jpg"])
    train_img_path = feat.vari()