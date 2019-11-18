# # Obrazek
import cv2 as cv
import numpy as np
# import skimage as ski
# from skimage import data, io, img_as_float
# import numpy as np
# import matplotlib.pyplot as plt
# # %matplotlib inline

# plt.figure(figsize=(20, 20))
# x = plt.subplot(2, 1, 1)
# x.axis('off')
# img = cv.imread("img/field.jpg")
# img = cv.resize(img, (700, 500))
# cv.imshow('Start', img)
# cv.moveWindow('Start', 600, 100)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Główny program
import skimage as ski
from skimage import data, io, img_as_float
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
sp = plt.subplot(3, 1, 1)
sp.axis('off')

img = cv.imread('img/field.jpg')

#img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
# converting into hsv image
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


lowerLimit = []
upperLimit = []


# stare, ręcznie robione

'''colortable = []
colortable.append([230,200,100])
colortable.append([10,0,10])
colortable.append([230,130,80])
colortable.append([100,170,190])
colortable.append([250,170,160])
colortable.append([230,240,250])
colortable.append([255,255,255])
for colorbgr in colortable:
    color = np.uint8([[colorbgr]])
    hsvcolor = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    lowerLimit.append([hsvcolor[0][0][0] - 10, 100, 100])
    upperLimit.append([hsvcolor[0][0][0] + 10, 255, 255])
'''
# Manual HSV limits for colors, first two are colors of shirts of both teams
lowerLimit.append([16, 72, 164])
upperLimit.append([36, 204, 239])
lowerLimit.append([141, 73, 27])
upperLimit.append([179, 140, 63])
# Background
lowerLimit.append([0, 77, 241])
upperLimit.append([8, 109, 265])
lowerLimit.append([6, 156, 218])
upperLimit.append([13, 185, 258])
lowerLimit.append([92, 123, 167])
upperLimit.append([102, 215, 230])
lowerLimit.append([87, 9, 242])
upperLimit.append([179, 56, 265])
lowerLimit.append([171, 153, 128])
upperLimit.append([182, 197, 217])
lowerLimit.append([2, 36, 175])
upperLimit.append([104, 132, 265])

# Define a mask ranging from lower to uppper
mask = 0
# Combining masks
# lowerlimit[x] upperlimit[x]
# x = 0 || 1 => team colors
# x > 1 => background (hopefully won't be needed)
for i in range(0, 2):
    maskpart = cv.inRange(hsv, np.array(
        lowerLimit[i]), np.array(upperLimit[i]))
    mask += maskpart
# Do masking
res = cv.bitwise_and(img, img, mask=mask)
# convert to hsv to gray
res_bgr = cv.cvtColor(res, cv.COLOR_HSV2BGR)
res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)


cv.imshow('Drugie', cv.cvtColor(res, cv.COLOR_BGR2RGB))
# Defining a kernel to do morphological operation in threshold image to get better output
kernel = np.ones((13, 13), np.uint8)
thresh = cv.threshold(
    res_gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

sp = plt.subplot(3, 1, 2)
sp.axis('off')
cv.imshow('Trzecie', cv.cvtColor(thresh, cv.COLOR_BGR2RGB))

# find contours in threshold image
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


prev = 0
font = cv.FONT_HERSHEY_SIMPLEX

idx = 0
# Countour processing
for c in contours:
    x, y, w, h = cv.boundingRect(c)

    # Detect players
    if(h >= (1.1)*w):
        if(100 > w > 15 and 150 > h >= 30):
            idx = idx+1
            player_img = img[y:y+h, x:x+w]
            player_hsv = cv.cvtColor(player_img, cv.COLOR_BGR2HSV)
            # First set
            maskfirst = cv.inRange(player_hsv, np.array(
                lowerLimit[0]), np.array(upperLimit[0]))
            res1 = cv.bitwise_and(player_img, player_img, mask=maskfirst)
            res1 = cv.cvtColor(res1, cv.COLOR_HSV2BGR)
            res1 = cv.cvtColor(res1, cv.COLOR_BGR2GRAY)
            nzCount = cv.countNonZero(res1)
            # Second set
            masksecond = cv.inRange(player_hsv, np.array(
                lowerLimit[1]), np.array(upperLimit[1]))
            res2 = cv.bitwise_and(player_img, player_img, mask=masksecond)
            res2 = cv.cvtColor(res2, cv.COLOR_HSV2BGR)
            res2 = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
            nzCountred = cv.countNonZero(res2)

            if(nzCount >= 20):
                # Mark first
                cv.putText(img, 'Pierwsza', (x-2, y-2), font,
                           0.8, (255, 0, 0), 2, cv.LINE_AA)
                cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            else:
                pass
            if(nzCountred >= 20):
                # Mark second
                cv.putText(img, 'Druga', (x-2, y-2), font,
                           0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            else:
                pass
sp = plt.subplot(3, 1, 3)
sp.axis('off')
cv.imshow('Czwarte', cv.cvtColor(img, cv.COLOR_BGR2RGB))
cv.waitKey(0)
cv.destroyAllWindows()
