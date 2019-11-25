from court_detection import find_court, draw_court
from lines_detection import find_lines, draw_lines
import math
import cv2 as cv
import numpy as np

cap = cv.VideoCapture('./clips/whole.mp4')

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv.resize(img, (800, 500))

    court_contours, court_mask = find_court(img, True)
    floor_contours = find_court(img, False)
    lines = find_lines(img)
    img_result = draw_court(img, court_contours)
    img_result = draw_lines(img_result, lines)

    cv.imshow('res', img_result)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
