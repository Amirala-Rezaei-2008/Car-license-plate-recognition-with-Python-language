import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils
import easyocr

car_img = cv.imread(image path)
car_img_gray = cv.cvtColor(car_img, cv.COLOR_BGR2GRAY)
plt.imshow(cv.cvtColor(car_img_gray, cv.COLOR_BGR2RGB))

blateral_filtered = cv.bilateralFilter(car_img_gray, 11, 15, 15)
edges = cv.Canny(blateral_filtered, 30, 200)
plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))

countours = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
countours_refined = imutils.grab_contours(countours)
countours_sorted = sorted(countours_refined, key=cv.contourArea, reverse=True)[:4]

for contour in countours_sorted:
    contour_approx = cv.approxPolyDP(contour, 10, True)
    if len(contour_approx) == 4:
        plate_location = contour_approx
        break

plate_mask0 = np.zeros(car_img_gray.shape, np.uint8)
plate_mask = cv.drawContours(plate_mask0, [plate_location], 0, 255, -1)
plate_image = cv.bitwise_and(car_img, car_img, mask=plate_mask)
plt.imshow(cv.cvtColor(plate_image, cv.COLOR_BGR2RGB))

cv.imwrite("car.jpeg", plate_image)

(X, y) = np.where(plate_mask == 255)
(X1, y1) = (np.min(X), np.min(y))
(X2, y2) = (np.max(X), np.max(y))
cropped_image = car_img_gray[X1:X2+1, y1:y2+1]

reader = easyocr.Reader(['en'])

plate_info = reader.readtext(plate_image)
print(plate_info)

plate_text = plate_info[0][-2]
print(plate_text)
