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

    return contours, mask_end_2


def draw_court(img, contours):

    if not contours:
        return img
    contoursImage = img.copy()
    for i, contour in enumerate(contours):
        if cv.contourArea(contour) > 10000:
            cv.drawContours(contoursImage, contours, i,
                            (0, 255, 0), thickness=4)
    return contoursImage


if __name__ == "__main__":

    cap = cv.VideoCapture('./clips/cut.mp4')
    cv.namedWindow('1')
    cv.moveWindow('1', 0, 100)
    cv.namedWindow('2')
    cv.moveWindow('2', 600, 100)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv.resize(img, (800, 500))

        contours, mask = find_court(img, True)
        contoursImage = draw_court(img, contours)

        cv.imshow('1', mask)
        cv.imshow('2', contoursImage)
        # cv.imshow('st', shi_tomasi(np.uint8(img)))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
