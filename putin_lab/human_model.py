"""
人間を両肩とその間を結ぶ線でモデル化し、matplotlibで描画してみる
"""

import numpy as np
import matplotlib.pyplot as plt

"""
単位は全てmm
"""

log_right=[]
log_left=[]

def human(center,theta):
    """
    人間の骨格モデル
    center: 肩の位置[mm]
    theta: 胸の向き[deg](y軸方向を0度として反時計回り)
    """
    theta=theta/180*np.pi
    width=500 #肩幅
    right=center+np.array([width/2*np.cos(theta),width/2*np.sin(theta)]) # 右肩の座標
    left=center+np.array([-width/2*np.cos(theta),-width/2*np.sin(theta)]) # 右肩の座標
    line=np.array([[right[0],left[0]],[right[1],left[1]]])
    vector=np.array([-np.sin(theta),np.cos(theta)])
    """
    plt.plot(right[0],right[1],marker='.',markersize=15,color="r")
    plt.plot(left[0],left[1],marker='.',markersize=15,color="b")
    plt.plot(line[0],line[1],color="g")
    """
    return right,left,line,vector

def robot(h_right,h_left,r_center):
    right_distance=np.linalg.norm(h_right-r_center)
    left_distance=np.linalg.norm(h_left-r_center)

    log_right.append(right_distance)
    log_left.append(left_distance)
    
    print(right_distance,left_distance)


center=np.array([0.0,0.0])
r_center=np.array([0.0,100.0])

theta=0
r_theta=0

speed=500.0
r_speed=500.0


# 直進

for i in range(3):
    right,left,line,vector=human(center,theta)
    center+=(speed*vector)
    robot(right,left,r_center)




# 左折

for i in range(10):
    right,left,line,vector=human(center,theta)
    center+=(speed*vector)
    theta+=10
    robot(right,left,r_center)



#plt.xlim(-3000,3000)
#plt.ylim(-1000,5000)
#plt.show()

print(np.linspace(0,len(log_right),len(log_right)))
plt.plot(np.linspace(0,len(log_right),len(log_right)),log_right)
plt.plot(np.linspace(0,len(log_right),len(log_right)),log_left)

plt.show()