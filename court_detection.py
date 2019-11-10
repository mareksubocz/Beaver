import cv2 as cv
import numpy as np
from numpy.core.numeric import bitwise_not

img = cv.imread('img/field.jpg')
img = cv.resize(img, (800, 500))
# img = cv.bilateralFilter(img, 9, 75, 75)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Teal range
lower_teal = np.array([50, 40, 40])
upper_teal = np.array([100, 255, 255])

# Orange range
lower_orange = np.array([00, 60, 60])
upper_orange = np.array([15, 255, 255])

# Define a mask ranging from lower to uppper
mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
mask_teal = cv.inRange(hsv, lower_teal, upper_teal)

# caly parkiet
# mask_court = cv.bitwise_or(mask_teal, mask_orange)

# smoothing
kernel = np.ones((5, 5), np.uint8)
mask_orange = cv.dilate(mask_orange, kernel, iterations=5)
mask_orange = cv.erode(mask_orange, kernel, iterations=5)


cv.imshow('mask', mask_orange)

contoursImage = img.copy()
contours, hierarchy = cv.findContours(
    mask_orange.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

contours_areas = list(map(cv.contourArea, contours))
# largest contour is court contour
court_contour = [contours[contours_areas.index(max(contours_areas))]]

court_mask = np.zeros((500, 800))
cv.fillPoly(court_mask, pts=court_contour, color=(1))

court_mask = np.array(court_mask, dtype=np.uint8)
court_mask *= 255

canny_court_mask = cv.Canny(court_mask, 50, 150)
cv.imshow('canny_court_mask', canny_court_mask)

# drawing lines
lines = cv.HoughLines(canny_court_mask, 1, np.pi/180, 100)

print(lines)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(contoursImage, (x1, y1), (x2, y2), (0, 255, 0), 2)

contours, hierarchy = cv.findContours(
    court_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


cv.drawContours(contoursImage, contours, 0,
                (0, 0, 255), thickness=4)


cv.namedWindow("1", cv.WINDOW_AUTOSIZE)
cv.imshow('1', contoursImage)
cv.moveWindow('1', 200, 100)
cv.waitKey()
cv.destroyAllWindows()
