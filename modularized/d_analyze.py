import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

def evaluate(img,img_rec):
    """
    学習画像・正常画像・異常画像それぞれについて、
    ・元画像
    ・再構成画像
    ・画素値の偏差のヒストグラム
    を出力
    """
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))
    ax1.imshow(img, cmap='gray')
    ax1.set_title("original img")
    ax2.imshow(img_rec, cmap='gray')
    ax2.set_title("reconstructed img")
    diff=abs(img-img_rec)
    ax3.imshow(diff*255,cmap='gray')
    ax3.set_title("difference")
    ax4.hist(diff.reshape(-1,),bins=255,range=(0,255))
    ax4.set_title("histgram")
    save_title=str(datetime.datetime.now())
    plt.savefig(os.getcwd()+"/img_data/"+save_title+".png")
    print("average: ",np.average(diff))
    print("median: ",np.median(diff))
    print("variance: ",np.var(diff))
    return np.average(diff),np.median(diff),np.var(diff)