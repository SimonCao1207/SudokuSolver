from digit_classifier import *
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import os

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

def prepare(img):
    img = cv2.bitwise_not(img)
    h, w = 30, 30
    y, x = 10, 10

    crop_img = img[y:y + h, x:x + w]
    resized = cv2.resize(crop_img, (28, 28), interpolation=cv2.INTER_AREA)

    return resized

def predict(boxes):
    predictions = []
    i = 1
    path = r"C:\Users\PC\PycharmProjects\SudokuSolver\SudokuImage"
    for box in boxes:
        # Save and read the image
        img = box.copy()
        name = 'img_' + str(i) + ".jpg"
        cv2.imwrite(os.path.join(path, name), img)
        img = cv2.imread(os.path.join(path, name))
        i += 1

        img = prepare(img)
        copy_img = img.copy()

        # Denoise the image
        denoised_square = ndimage.median_filter(copy_img, 3)
        white_pix_count = np.count_nonzero(denoised_square)

        # Detech empty box
        if white_pix_count > 100:
            empty_square = False
        else:
            empty_square = True
        if empty_square:
            predictions.append(-1)
            continue

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        im_pil = Image.fromarray(blackAndWhiteImage)
        imtest = transforms.ToTensor()(im_pil).unsqueeze_(0)
        num = thresholding(model(imtest))
        predictions.append(num.detach().numpy()[0])

    return predictions


