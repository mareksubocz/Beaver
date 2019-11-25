import cv2 as cv
import numpy as np
import math
from statistics import mean
from court_detection import find_court
from random import randint


def find_lines(img):
    _, court_mask = find_court(img, True)

    # * smoothing
    kernel = np.ones((5, 5), np.uint8)
    court_mask = cv.erode(court_mask, kernel, iterations=1)

    img2 = img.copy()
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    threshold, img2 = cv.threshold(img2, 200, 255, cv.THRESH_BINARY)

    img2 = np.array(img2, dtype=np.float64)
    result_mask = cv.bitwise_and(img2, court_mask)
    result_mask = np.array(result_mask, dtype=np.uint8)

    linesP = cv.HoughLinesP(result_mask, 1,
                            np.pi/180, 50, None, 50, 5)

    if linesP is None:
        return None

    return [x[0] for x in linesP]


def draw_lines(img_original, lines):
    if lines is None:
        return img_original
    img = img_original.copy()
    for i in range(0, len(lines)):
        x1, y1, x2, y2 = lines[i]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        if 10 > angle > -10:
            continue
        cv.line(img, (x1, y1), (x2, y2),
                (0, 0, 255), 3, cv.LINE_AA)
    return img


if __name__ == "__main__":
    cap = cv.VideoCapture('./clips/cut.mp4')
    cv.namedWindow('1')
    cv.moveWindow('1', 300, 100)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv.resize(img, (800, 500))

        lines = find_lines(img)
        img = draw_lines(img, lines)

        cv.imshow('1', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
