import numpy as np
import cv2
from cv2 import aruco
import sys

def main(args):
    """
    引数0→写真
    引数1→動画
    """
    cap = cv2.VideoCapture(0)
    if args == "0":
        ret,img = cap.read()
        cv2.imwrite('test.jpg', img)
        
    else:
        while True:
            ret, img = cap.read()
            cv2.imshow('drawDetectedMarkers', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = sys.argv
    main(args[1])
