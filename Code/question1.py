import cv2
import numpy as np
import matplotlib.pyplot as plt
#read the image
img = cv2.imread("Q1image.png")

circular_kernal = [[0,0,1,0,0] ,[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]]
# cir = np.array(circular_kernal)
cir =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
#erosion to seperate individual pieces
kernal = np.ones((7,7),dtype = np.uint8)
img2 = cv2.erode(img,cir,iterations = 3)

#closing operation to remove unnecessary white holes
kernel = np.ones((5,5) , dtype = np.uint8)
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

#gray scaling to detect contours
imgray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
th, threshold = cv2.threshold(imgray, 100,255,0)

#find and draw contours
cnts= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
img3 = cv2.drawContours(img2,cnts,-1,(0,255,0),3)

#count countours
s1 = 230
s2 = 585
xcnts = []
for cnt in cnts:
    if s1<cv2.contourArea(cnt)<s2:
        xcnts.append(cnt)
print('Number of circles: ' ,len(cnts))

kernel = np.ones((11,11),np.uint8)
dilation = cv2.dilate(imgray,cir,iterations = 2)

#printing the number on coins
sorted_contours = sorted(cnts, key = cv2.contourArea )
color = cv2.cvtColor(dilation,cv2.COLOR_GRAY2RGB)
color = cv2.GaussianBlur(color ,(3,3) ,cv2.BORDER_DEFAULT )
for i,c in enumerate(sorted_contours):
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    cv2.putText(color , text = str(i+1) , org=(cx,cy) , fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(240,100,240),
            thickness=2, lineType=cv2.LINE_AA)
plt.imshow(dilation,cmap = 'gray')
plt.show()

fig = plt.figure(figsize=(6,6))
rows = 3
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Original Image")

fig.add_subplot(rows, columns, 2)
plt.imshow(img2)
plt.axis('off')
plt.title("Erosion")

fig.add_subplot(rows, columns, 3)
plt.imshow(closing)
plt.axis('off')
plt.title("Closing")

fig.add_subplot(rows, columns, 4)
plt.imshow(color)
plt.axis('off')
plt.title("Counted")

fig.add_subplot(rows, columns, 4)
plt.imshow(color)
plt.axis('off')
plt.title("Counted")

fig.add_subplot(rows, columns, 5)
plt.imshow(dilation ,cmap = 'gray')
plt.axis('off')
plt.title("Dilated final")
plt.show()



