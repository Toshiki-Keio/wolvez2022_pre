import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco = cv2.aruco.drawMarker(aruco_dict, 0, 256)
cv2.imwrite("aruco.png", aruco)