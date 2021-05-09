from utils import *
from digit_classifier import *
import os

PATH = 'model.pth'
pathImage = "SudokuImage/sudoku_img2.png"
heightImage = 450
widthImage = 450
model = clf

# Prepare image
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImage, heightImage))
imgThreshold = preProcess(img)

#Find contours
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 3) # draw all the contours

biggest, max_area = biggestContour(contours)

if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImage, 0], [0, heightImage], [widthImage, heightImage]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # Transformation matrix
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImage, heightImage))
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_RGB2GRAY)

    # SPLIT the image into boxes of number
    boxes = split_boxes(imgWarpColored)
    imgtest = boxes[0].copy()
    directory = r'C:\Users\PC\PycharmProjects\SudokuSolver'
    os.chdir(directory)
    cv2.imwrite("number3.png", imgtest)
    print(imgtest.shape)

cv2.imshow('Sudoku Board', imgThreshold)
cv2.waitKey(0)