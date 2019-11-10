import cv2 as cv
import numpy as np
from numpy.core.numeric import bitwise_not
from time import sleep

img = cv.imread('img/field.jpg')
cap = cv.VideoCapture('./voleybol.mp4')
# vidObj = cv.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv.resize(img, (800, 500))
    # img = cv.bilateralFilter(img, 9, 75, 75)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Teal range
    lower_teal = np.array([50, 40, 40])
    upper_teal = np.array([100, 255, 255])

    # Orange range
    lower_orange = np.array([0, 120, 120])
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

    # cv.imshow('mask', mask_orange)

    contoursImage = img.copy()
    contours, hierarchy = cv.findContours(
        mask_orange.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours_areas = list(map(cv.contourArea, contours))
    # largest contour is court contour
    if contours_areas:
        court_contour = [contours[contours_areas.index(max(contours_areas))]]

        court_mask = np.zeros((500, 800))
        cv.fillPoly(court_mask, pts=court_contour, color=(1))

        court_mask = np.array(court_mask, dtype=np.uint8)
        court_mask *= 255

        canny_court_mask = cv.Canny(court_mask, 50, 150)
        # cv.imshow('canny_court_mask', canny_court_mask)

        contours, hierarchy = cv.findContours(
            court_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        hull = []
        for contour in contours:
            hull.append(cv.convexHull(contour, False))

        if cv.contourArea(hull[0]) > 40000:
            cv.drawContours(contoursImage, hull, 0,
                            (0, 0, 255), thickness=4)

    cv.imshow('1', contoursImage)
    cv.moveWindow('1', 200, 100)
    # sleep(0.1)
    # cv.waitKey()
    # cv.namedWindow("1", cv.WINDOW_AUTOSIZE)
    # cv.destroyAllWindows()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
