import numpy as np
import cv2
from matplotlib import pyplot as plt

# 分類器の指定
# cascade = cv2.CascadeClassifier("/Users/toshikifukui/Downloads/opencv-4.3.0/data/haarcascades/haarcascade_eye.xml")
cascade = cv2.CascadeClassifier("/Users/toshikifukui/Downloads/opencv-4.3.0/data/haarcascades/haarcascade_frontalface_alt2.xml")
# read images
imgR = cv2.imread('/Users/toshikifukui/Desktop/image.png')
imgL = cv2.imread('/Users/toshikifukui/Desktop/image2.png')

imgR = np.array(imgR)
imgL = np.array(imgL)
#グレースケール変換
#物体認識（顔認識）の実行
facerect = cascade.detectMultiScale(imgL)

#print(facerect)


# 検出した場合
#物体認識（顔認識）の実行
facerect = cascade.detectMultiScale(imgR)
if len(facerect) > 0:
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        cv2.rectangle(imgR, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0,255,255), 2)
        left_up = rect[0:2]
        right_down = rect[0:2]+rect[2:4]
        print(left_up)
        print(right_down)
        area = abs(left_up[0]-right_down[0])*abs(left_up[1]-right_down[1])
        print(area)
        middle_R = [(left_up[0]+right_down[0])/2 , (left_up[1]+right_down[1])/2]
        print("midde_R:",middle_R)

    cv2.imwrite("output.jpg", imgR)
    # cv2.imshow("output",imgR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#物体認識（顔認識）の実行
facerect = cascade.detectMultiScale(imgL)
if len(facerect) > 0:
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        cv2.rectangle(imgL, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0,255,255), 2)
        left_up = rect[0:2]
        right_down = rect[0:2]+rect[2:4]
        middle_L = [(left_up[0]+right_down[0])/2 , (left_up[1]+right_down[1])/2]
        print("middle_L:",middle_L)

    #認識結果の保存
    # cv2.imwrite("output.jpg", imgL)
    # cv2.imshow("output",imgR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

import numpy as np

"""
センサ情報
3.674*2.760mm
3296*2512pixel
→
横: 3.674/3296*1e-3m
縦: 2.760/2512*1e-3m


"""

point_L=np.array(middle_L)
point_R=np.array(middle_R)
shisa_gaso=point_R-point_L

m_per_gaso_x=3.674/3296*1e-3
m_per_gaso_y=2.760/2512*1e-3

s=np.sqrt((shisa_gaso[0]*m_per_gaso_x)**2+(shisa_gaso[1]*m_per_gaso_y)**2)
c=0.12

f=3.04*1e-3

d=f*c/s
print(f"視差s': {shisa_gaso} 画素")
print(f"視差s: {s} m")
print(f"カメラ間の距離c: {c} m")
print(f"焦点距離: {f} m")
print(f"深さd: {d} m")

"""
視差s 10pixel=11um=11e-3mm
焦点距離f 3mm
カメラ間c 120mm
d:c=f:s
d:120=3:0.011
d=360/0.011
"""
