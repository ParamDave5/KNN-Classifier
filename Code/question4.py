import cv2
from cv2 import mean
import matplotlib.pyplot as plt
import numpy as np

def read(path):
    img = cv2.imread(path)
    return img

def Distance(x1,x2):
    dst = np.sqrt(np.sum((x1-x2)**2))
    return dst

img = read('Q4image.png')
image = np.copy(img)
image = np.float32(image)

K = 4
print(np.shape(image))
index1 = np.random.choice(image.shape[0],K,replace=False)
index2 = np.random.choice(image.shape[1],K,replace=False)

index = []

for i in range(len(index1)):
    index.append([index1[i],index2[i]])

meanas = []

for i in index:
    meanas.append(image[i[0]][i[1]])

meanas = np.array(meanas)

iterations = 0
clusters = []

while True:
    iterations = iterations + 1    
    clusters = [[],[],[],[]]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            d1 = Distance(image[i][j],meanas[0])
            d2 = Distance(image[i][j],meanas[1])
            d3 = Distance(image[i][j],meanas[2])
            d4 = Distance(image[i][j],meanas[3])
            distances = [d1,d2,d3,d4]
            clusters[np.argmin(distances)].append([image[i][j],(i,j)])

    
    clusters[0] = np.array(clusters[0])
    clusters[1] = np.array(clusters[1])
    clusters[2] = np.array(clusters[2])
    clusters[3] = np.array(clusters[3])
    
    old_meanas = np.copy(meanas)
    meanas[0] = np.mean(clusters[0][:,0],axis=0)
    meanas[1] = np.mean(clusters[1][:,0],axis=0)
    meanas[2] = np.mean(clusters[2][:,0],axis=0)
    meanas[3] = np.mean(clusters[3][:,0],axis=0)

    d1 = Distance(old_meanas[0],meanas[0])
    d2 = Distance(old_meanas[1],meanas[1])
    d3 = Distance(old_meanas[2],meanas[2])
    d4 = Distance(old_meanas[3],meanas[3])
    distances = [d1,d2,d3,d4]
    print(old_meanas)
    print(meanas)

    print(np.sum(distances))
    if np.sum(distances) < 0.4:
        break
    

meanas = meanas.astype(int)
print(meanas)
for i in range(len(clusters)):
    for j in range(len(clusters[i])):
        index = clusters[i][j][1]
        img[index[0]][index[1]][0] = meanas[i][0]
        img[index[0]][index[1]][1] = meanas[i][1]
        img[index[0]][index[1]][2] = meanas[i][2]
plt.imshow(cv2.cvtColor(img ,cv2.COLOR_BGR2RGB))
plt.show()


