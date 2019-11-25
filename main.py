from court_detection import shi_tomasi
from court_detection import find_court
from marking import markBall, markPlayers
from test import find_mask
import math
import cv2 as cv
import numpy as np

cap = cv.VideoCapture('./clips/cut.mp4')

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv.resize(img, (800, 500))

    # Teal range
    lower_teal = np.array([50, 10, 40])
    upper_teal = np.array([100, 255, 255])

    # Orange
    lower_orange = np.array([0, 50, 80])
    upper_orange = np.array([20, 255, 255])
    # Light orange
    lower_lorange = np.array([170, 30, 20])
    upper_lorange = np.array([180, 255, 255])

    lower_white = np.array([0, 30, 180])
    upper_white = np.array([180, 50, 255])

    mask_teal = find_mask(img, lower_teal, upper_teal)
    mask_orange = find_mask(img, lower_orange, upper_orange)
    mask_lorange = find_mask(img, lower_lorange, upper_lorange)
    mask_white = find_mask(img, lower_white, upper_white)
    mask = cv.bitwise_or(mask_orange, mask_teal)
    mask = cv.bitwise_or(mask, mask_lorange)

    court_contours = find_court(img, True)
    floor_contours = find_court(img, False)

    contour_image = img.copy()
    for i in range(len(court_contours)):
        cv.drawContours(img, court_contours, i, (0, 0, 255), thickness=3)

    cv.imshow('mask', mask)
    cv.imshow('players', markPlayers(mask, court_contours, img))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
