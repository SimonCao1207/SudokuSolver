import cv2
import numpy as np
import torch
from torchvision import models

# preprocessing image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

# find corners in a Sudoku board
def biggestContour(contours):
    biggest =  np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                max_area = area
                biggest = approx
    return biggest, max_area

# reorder points for warp perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    newPoints = np.zeros((4,1,2), dtype=np.int32)
    add = myPoints.sum(1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

# Split the image into boxes
def split_boxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for col in cols:
            boxes.append(col)
    return boxes

# Digit Recognition model
# def getPrediction(boxes, model):
#     result = []
#     for image in boxes:
#         ## PREPARE IMAGE
#         img = np.asarray(image)
#         img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
#         img = cv2.resize(img, (28, 28))
#         img = img / 255
#         img = img.reshape(1, 28, 28, 1)
#         ## GET PREDICTION
#         predictions = model(img)
#         classIndex = model.predict_classes(img)
#         probabilityValue = np.amax(predictions)
#         ## SAVE TO RESULT
#         if probabilityValue > 0.8:
#             result.append(classIndex[0])
#         else:
#             result.append(0)
#     return result

