import numpy as np
import cv2 as cv

# # Teal range
# lower_teal = np.array([50, 10, 40])
# upper_teal = np.array([100, 255, 255])

# # Orange
# lower_orange = np.array([0, 50, 80])
# upper_orange = np.array([20, 255, 255])
# # Light orange
# lower_lorange = np.array([170, 30, 20])
# upper_lorange = np.array([180, 255, 255])

# # Define a mask ranging from lower to uppper
# mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
# mask_lorange = cv.inRange(hsv, lower_lorange, upper_lorange)
# mask_teal = cv.inRange(hsv, lower_teal, upper_teal)
# mask = mask_orange
# # mask = cv.bitwise_or(mask_orange, mask_lorange)
# mask = cv.bitwise_or(mask, mask_teal)

# # mask =


def find_mask(img, lower_color, upper_color):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_color, upper_color)

    # # Do masking
    res = cv.bitwise_and(img, img, mask=mask)
    # convert to hsv to gray
    res_bgr = cv.cvtColor(res, cv.COLOR_HSV2BGR)
    res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    # image to get better output.
    # Defining a kernel to do morphological operation in threshold
    # kernel = np.ones((13, 13), np.uint8)
    # thresh = cv.threshold(res_gray, 127, 255,
    #                       cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # thresh = cv.bitwise_not(thresh)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=5)
    mask = cv.dilate(mask, kernel, iterations=5)

    return mask
