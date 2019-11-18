from court_detection import find_court
from marking import markBall, markPlayers
import cv2 as cv
import numpy as np

cap = cv.VideoCapture('./clips/cut.mp4')

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv.resize(img, (800, 500))

    court_contours, court_mask = find_court(img, True)
    floor_contours, floor_mask = find_court(img, False)

    cv.imshow('ball', markBall(floor_mask, img))
    cv.imshow('players', markPlayers(floor_mask, court_contours, img))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
