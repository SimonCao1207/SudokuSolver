from digit_classifier import load_model, get_pred
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import os
from tqdm import tqdm


def display(img, name='board'): 
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preProcess(img):
    """
    Preprocessing the image
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

def biggestContour(contours):
    """ 
    Find corners in a Sudoku board
    """
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

def reorder(myPoints):
    """
    Reorder points for warp perspective
    """
    myPoints = myPoints.reshape((4, 2))
    newPoints = np.zeros((4,1,2), dtype=np.int32)
    add = myPoints.sum(1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

def split_boxes(img):
    """
    Split the image into boxes
    """
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for col in cols:
            boxes.append(col)
    return boxes

def resize(img):
    img = cv2.bitwise_not(img)
    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    return resized

def preprocess(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize(28)
    ])
    img = transform(img)
    return img

def print_grid(grid):
    for r in grid:
        print(r)
        
def getImage(path, num, box):
    cv2.imwrite(os.path.join(path, f"{num}.png"), box)

def crop(img, c):
    h, w = img.shape
    return img[c:h-c, c:w-c]

def isWhite(img, thres=90):
    img = crop(img, 10)
    img = cv2.bitwise_not(img)
    _, blackAndWhite = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    white_pix_count = np.count_nonzero(blackAndWhite)
    if white_pix_count > thres:
        return False
    return True