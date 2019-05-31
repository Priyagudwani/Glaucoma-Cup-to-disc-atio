from skimage.io import imread
from skimage.filters import threshold_otsu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

imgg = cv2.imread('gl_2.jpg')
imgg = cv2.resize(imgg,(600,400))


def area_of_circle(r):
    """Function that defines an area of a circle"""
    a = r**2 * math.pi
    return a


img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
clone = img_gray.copy()
kernel = np.ones((5,5),np.uint8)
kernel1 = np.ones((15,15),np.uint8)
kernel3 = np.ones((25,25),np.uint8)

fig, (ax1, ax2 , ax3) = plt.subplots(3, 1)

cv2.imshow('grey',img_gray)
ax1.imshow(img_gray, cmap="gray")


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))   #adaptive histogram
cl1 = clahe.apply(img_gray)


equ = cv2.equalizeHist(img_gray)
res = np.hstack((img_gray,equ)) #stacking images side-by-side
   
    

img_m = cv2.medianBlur(cl1 ,5)    #median blur
ax2.imshow(img_m, cmap="gray")


equ = cv2.equalizeHist(img_m)
res = np.hstack((img_gray,equ)) #stacking images sde-by-side

blur = cv2.GaussianBlur(cl1,(5,5),8)  # blurring filter

th3 = cv2.adaptiveThreshold(img_m,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,2)  # vessels extraction


closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

dilation = cv2.dilate(blur,kernel3,iterations = 1)

closing1 = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel3)

circles = cv2.HoughCircles(closing1,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=10,maxRadius=100)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(imgg ,(i[0],i[1]),i[2],(0,255,0),1)
    print("disc")
    
    a_disc=area_of_circle(i[2])
    print(a_disc)
    # draw the center of the circle
    cv2.circle(imgg ,(i[0],i[1]),2,(0,0,255),2)

cv2.imshow('detected optic disc',imgg )


(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_gray)
cv2.circle(imgg,maxLoc,1,(0,0,255),2)  # draw the center of the circle

rows,cols = maxLoc

clone = closing.copy()

x1=rows-65
y1=cols-55
x2=rows+65
y2=cols+85

cv2.rectangle(imgg, (x1, y1), (x2, y2), (0,0,255), 1)

frame = clone[y1:y2, x1:x2]
img = cv2.imread("kl.png")

img_o=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER ,20, 1.0) #+ cv2.TERM_CRITERIA_MAX_ITER+ cv2.TERM_CRITERIA_EPS

ret,label,center=cv2.kmeans(Z,7,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.ravel()]
res2 = res.reshape((img.shape))


cv2.imshow("clustering", res2)

close = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel)

img = cv2.medianBlur(close,15)

(thresh, im_bw) = cv2.threshold(img,close.max()-30,close.max(),0)
ot_bw=cv2.cvtColor(im_bw, cv2.COLOR_BGR2GRAY)
cv2.imshow("threshold", ot_bw) 


im2, contours, hierarchy = cv2.findContours(ot_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(img,center,radius,(0,255,0),1)
    a_cup=area_of_circle(radius)
    if a_cup > 500:
        aa_cup=a_cup
    print("cup")
  

cv2.imshow("bbb",img)
print(a_cup)
print("ratio",a_cup/a_disc )


cv2.waitKey(0)
cv2.destroyAllWindows()
