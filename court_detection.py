import math
from random import randint
from time import sleep

import cv2 as cv
import numpy as np


def shi_tomasi(image):

    # Converting to grayscale
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners
    corners_img = cv.goodFeaturesToTrack(gray_img, 1000, 0.01, 10)

    corners_img = np.int0(corners_img)

    for corners in corners_img:

        x, y = corners.ravel()
        # Circling the corners in green
        cv.circle(image, (x, y), 3, [0, 255, 0], -1)

    return image


def find_court(img, just_court: bool):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Teal range
    lower_teal = np.array([50, 10, 40])
    upper_teal = np.array([100, 255, 255])
    # Orange
    lower_orange = np.array([0, 50, 80])
    upper_orange = np.array([20, 255, 255])
    # Light orange
    lower_lorange = np.array([170, 30, 20])
    upper_lorange = np.array([180, 255, 255])

    # Define a mask ranging from lower to uppper
    mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
    mask_lorange = cv.inRange(hsv, lower_lorange, upper_lorange)
    mask_teal = cv.inRange(hsv, lower_teal, upper_teal)

    mask_sum_orange = cv.bitwise_or(mask_orange, mask_lorange)

    # caly parkiet
    if not just_court:
        mask_sum_orange = cv.bitwise_or(mask_teal, mask_orange)

    # * smoothing
    kernel = np.ones((5, 5), np.uint8)
    mask_sum_orange = cv.erode(mask_sum_orange, kernel, iterations=2)
    mask_sum_orange = cv.dilate(mask_sum_orange, kernel, iterations=4)
    mask_sum_orange = cv.erode(mask_sum_orange, kernel, iterations=2)

    canny_orange = cv.Canny(mask_sum_orange, 50, 150)

    # * contours
    contours, hierarchy = cv.findContours(
        mask_sum_orange.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return contours
    areas = [cv.contourArea(c) for c in contours]
    max_index = areas.index(max(areas))
    # contours[max_index] = cv.convexHull(contours[max_index], False)
    mask_end = np.zeros((500, 800))
    cv.fillPoly(mask_end, pts=[contours[max_index]], color=(1))

    kernel = np.ones((7, 7), np.uint8)
    mask_end = cv.dilate(mask_end, kernel, iterations=8)
    mask_end = cv.erode(mask_end, kernel, iterations=8)
    # mask_end = cv.dilate(mask_end, kernel, iterations=5)

    mask_end = cv.threshold(mask_end, 0, 255, cv.THRESH_BINARY)[1]
    mask_end_2 = mask_end
    mask_end = np.uint8(mask_end)

    contours, hierarchy = cv.findContours(
        mask_end.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours, mask_end


if __name__ == "__main__":

    cap = cv.VideoCapture('./cut.mp4')
    # cap = cv.VideoCapture(0)
    cv.namedWindow('1')
    cv.moveWindow('1', 0, 100)
    cv.namedWindow('2')
    cv.moveWindow('2', 600, 100)

    while True:
        success, img = cap.read()

        if not success:
            break
        # img = cv.imread('img/field.jpg')
        img = cv.resize(img, (800, 500))

        img2 = img.copy()
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        ret, img2 = cv.threshold(img2, 200, 255, cv.THRESH_TOZERO)
        # img2 = cv.Canny(img2, 50, 150)
        # * HoughLines
        lines = cv.HoughLines(img2, 1, np.pi / 180, 150)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(img, pt1, pt2, (255, 0, 255), 3, cv.LINE_AA)

        # * PHoughLines
        # linesP = cv.HoughLinesP(img2, 1, np.pi / 180, 50, None, 300, 30)
        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         cv.line(img, (l[0], l[1]), (l[2], l[3]),
        #                 (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow('img2', img2)

        # img = cv.bilateralFilter(img, 9, 75, 75)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Teal range
        lower_teal = np.array([50, 10, 40])
        upper_teal = np.array([100, 255, 255])

        # Orange
        lower_orange = np.array([0, 50, 80])
        upper_orange = np.array([20, 255, 255])
        # Light orange
        lower_lorange = np.array([170, 30, 20])
        upper_lorange = np.array([180, 255, 255])

        # Define a mask ranging from lower to uppper
        mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
        mask_lorange = cv.inRange(hsv, lower_lorange, upper_lorange)
        mask_teal = cv.inRange(hsv, lower_teal, upper_teal)

        mask_sum_orange = cv.bitwise_or(mask_orange, mask_lorange)

        # caly parkiet
        # mask_orange = cv.bitwise_or(mask_teal, mask_orange)

        # * smoothing
        kernel = np.ones((5, 5), np.uint8)
        mask_sum_orange = cv.erode(mask_sum_orange, kernel, iterations=2)
        mask_sum_orange = cv.dilate(mask_sum_orange, kernel, iterations=4)
        mask_sum_orange = cv.erode(mask_sum_orange, kernel, iterations=2)
        # mask_sum_orange = cv.dilate(mask_sum_orange, kernel, iterations=2)

        canny_orange = cv.Canny(mask_sum_orange, 50, 150)
        # cv.imshow('mask', mask_orange)

        # * contours
        contoursImage = img.copy()
        contours, hierarchy = cv.findContours(
            mask_sum_orange.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue
        areas = [cv.contourArea(c) for c in contours]
        max_index = areas.index(max(areas))
        # contours[max_index] = cv.convexHull(contours[max_index], False)
        mask_end = np.zeros((500, 800))
        cv.fillPoly(mask_end, pts=[contours[max_index]], color=(1))

        kernel = np.ones((7, 7), np.uint8)
        mask_end = cv.dilate(mask_end, kernel, iterations=8)
        mask_end = cv.erode(mask_end, kernel, iterations=8)
        # mask_end = cv.dilate(mask_end, kernel, iterations=5)

        mask_end = cv.threshold(mask_end, 0, 255, cv.THRESH_BINARY)[1]
        mask_end_2 = mask_end
        mask_end = np.uint8(mask_end)

        contours, hierarchy = cv.findContours(
            mask_end.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        for i, contour in enumerate(contours):
            if cv.contourArea(contour) > 10000:
                cv.drawContours(contoursImage, contours, i,
                                (0, 255, 0), thickness=4)
        # for i, contour in enumerate(contours):
        #     cv.drawContours(contoursImage, contours, i,
        #                     (0, 255, 0), thickness=4)

        # # Do masking
        # res = cv.bitwise_and(img, img, mask=mask_orange)
        # # convert to hsv to gray
        # res_bgr = cv.cvtColor(res, cv.COLOR_HSV2BGR)
        # res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        # # Defining a kernel to do morphological operation in threshold #image to get better output.
        # kernel = np.ones((13, 13), np.uint8)
        # thresh = cv.threshold(
        #     res_gray, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        # thresh = cv.morphologyEx(mask_orange, cv.MORPH_CLOSE, kernel)

        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 100)
        # if circles is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles = np.round(circles[0, :]).astype("int")

        #     for (x, y, r) in circles:
        #         if r > 10:
        #             continue
        #         cv.circle(contoursImage, (x, y), r, (0, 0, 255), 4)
        #         cv.rectangle(contoursImage, (x - 5, y - 5),
        #                      (x + 5, y + 5), (0, 128, 255), -1)

        cv.imshow('1', mask_end_2)
        cv.imshow('2', contoursImage)
        # cv.imshow('st', shi_tomasi(np.uint8(img)))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
