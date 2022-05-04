import cv2 as cv
dirname="img_data"
filename="img_test_ng.jpg"
path=dirname+"/"+filename
img = cv.imread(path)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray=cv.Canny(img_gray,100,200)
print(path[:-4])
cv.imwrite(path[:-4]+"_edge.jpg",gray)