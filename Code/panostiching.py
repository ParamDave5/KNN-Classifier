import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

#read images from the path
def read_image(path):
    img = cv2.imread(path)
    return img

#convert images to gray scale
def gray_scale(img):
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return gray

#describe freatures using ORB method
def featureDescripter(image):
    descriptor = cv2.ORB_create()
    (keyPoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keyPoints, features)

#matching key feature points
def matchFeaturesBF(featuresA, featuresB, method):
    bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    best_matches = bruteForce.match(featuresA,featuresB)
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

#find homograpy from keypoints 
def Homography(keyPointsA, keyPointsB, featuresA, featuresB, matches, homoThresh):
    keyPointsA = np.float32([kp.pt for kp in keyPointsA])
    keyPointsB = np.float32([kp.pt for kp in keyPointsB])
    
    if len(matches) > 4:

        ptsA = np.float32([keyPointsA[m.queryIdx] for m in matches])
        ptsB = np.float32([keyPointsB[m.trainIdx] for m in matches])
        
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            homoThresh)

        return (matches, H, status)
    else:
        return None


img_2 = read_image('Q2imageA.png')
img_1 = read_image('Q2imageB.png')

img1 = gray_scale(img_1)
img2 = gray_scale(img_2)

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()

keyPointsA, featuresA = featureDescripter(img1)
keyPointsB, featuresB = featureDescripter(img2)


fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
ax1.imshow(cv2.drawKeypoints(img1,keyPointsA,None,color=(0,255,0)))
ax1.set_xlabel("(a)", fontsize=14)
ax2.imshow(cv2.drawKeypoints(img2,keyPointsB,None,color=(0,255,0)))
ax2.set_xlabel("(b)", fontsize=14)
plt.show()


fig = plt.figure(figsize=(20,8))

matches = matchFeaturesBF(featuresA, featuresB, method='orb')
img3 = cv2.drawMatches(img1,keyPointsA,img2,keyPointsB,matches[:100], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.show()

M = Homography(keyPointsA, keyPointsB, featuresA, featuresB, matches, homoThresh=4)
if M is None:
    print("Empty None!")
(matches, H, status) = M
print(H)

width = img1.shape[1] + img2.shape[1]
height = img1.shape[0] + img2.shape[0]

result = cv2.warpPerspective(img1, H, (width, height))
result[0:img2.shape[0], 0:img2.shape[1]] = img2


plt.imshow(result)
plt.axis('off')
plt.show()
final = result[0:340,0:630]

plt.imshow(final)
plt.show()






    


