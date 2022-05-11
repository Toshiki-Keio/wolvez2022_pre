import numpy as np
import cv2
from cv2 import aruco

def main():
    cap = cv2.VideoCapture(0)
    # マーカーサイズ
    marker_length = 0.056 # [m]
    # マーカーの辞書選択
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    camera_matrix = np.load("camera/mtx.npy")
    distortion_coeff = np.load("camera/dist.npy")

    while True:
        ret, img = cap.read()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
        # 可視化
        aruco.drawDetectedMarkers(img, corners, ids, (0,255,255))

        if len(corners) > 0:
            # マーカーごとに処理
            for i, corner in enumerate(corners):
                # rvec -> rotation vector, tvec -> translation vector
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coeff)

                # < rodoriguesからeuluerへの変換 >

                # 不要なaxisを除去
                tvec = np.squeeze(tvec)
                rvec = np.squeeze(rvec)
                # 回転ベクトルからrodoriguesへ変換
                rvec_matrix = cv2.Rodrigues(rvec)
                rvec_matrix = rvec_matrix[0] # rodoriguesから抜き出し
                # 並進ベクトルの転置
                transpose_tvec = tvec[np.newaxis, :].T
                # 合成
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                # オイラー角への変換
                euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]

                print("x : " + str(tvec[0]))
                print("y : " + str(tvec[1]))
                print("z : " + str(tvec[2]))
                print("roll : " + str(euler_angle[0]))
                print("pitch: " + str(euler_angle[1]))
                print("yaw  : " + str(euler_angle[2]))

                # 可視化
                draw_pole_length = marker_length/2 # 現実での長さ[m]
                aruco.drawAxis(img, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)

        cv2.imshow('drawDetectedMarkers', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from mpl_toolkits.mplot3d import axes3d, Axes3D

# def plot_all_frames(elev=90, azim=270):
#     frames = []

#     for t in tqdm(range(len(XYZ))):
#         fig = plt.figure(figsize=(4,3))
#         ax = Axes3D(fig)
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2, 2)
#         ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

#         x, y, z = XYZ[t]
#         ux, vx, wx = V_x[t]
#         uy, vy, wy = V_y[t]
#         uz, vz, wz = V_z[t]

#         # draw marker
#         ax.scatter(0, 0, 0, color="k")
#         ax.quiver(0, 0, 0, 1, 0, 0, length=1, color="r")
#         ax.quiver(0, 0, 0, 0, 1, 0, length=1, color="g")
#         ax.quiver(0, 0, 0, 0, 0, 1, length=1, color="b")
#         ax.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1], [0,0,0,0,0], color="k", linestyle=":")

#         # draw camera
#         ax.quiver(x, y, z, ux, vx, wx, length=0.5, color="r")
#         ax.quiver(x, y, z, uy, vy, wy, length=0.5, color="g")
#         ax.quiver(x, y, z, uz, vz, wz, length=0.5, color="b")

#         # save for animation
#         fig.canvas.draw()
#         frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
#         plt.close()

#     return frames

# marker_length = 0.07 # [m] ### 注意！
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# mtx = np.load("camera/mtx.npy")
# dist = np.load("camera/dist.npy")
# # VideoCapture オブジェクトを取得します
# capture = cv2.VideoCapture(0)
# XYZ = []
# RPY = []
# V_x = []
# V_y = []
# V_z = []

# while True:
#     ret, frame = capture.read()
#     # cv2.imshow('frame',frame)
#     corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

#     if len(corners) == 0:
#         continue

#     rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

#     R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列
#     R_T = R.T
#     T = tvec[0].T

#     xyz = np.dot(R_T, - T).squeeze()
#     XYZ.append(xyz)

#     rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])
#     RPY.append(rpy)

#     V_x.append(np.dot(R_T, np.array([1,0,0])))
#     V_y.append(np.dot(R_T, np.array([0,1,0])))
#     V_z.append(np.dot(R_T, np.array([0,0,1])))

#     # ---- 描画
#     cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0,255,255))
#     cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, marker_length/2)
#     cv2.imshow('frame', frame)
#     plot_all_frames(elev=105, azim=270)
#     plot_all_frames(elev=165, azim=270)
#     cv2.waitKey(1)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()
