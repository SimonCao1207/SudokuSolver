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

def thres(img):
    """
    Preprocessing the image
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

def biggestContour(contours):
    """ 
    Find corners in the board
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

def warp(img, ps):
    """"
    Warp perspective image
    """

    if (len(img.shape) > 2):
        w, h, _ = img.shape
    else:  w, h = img.shape
    ps1 = np.float32(ps)
    ps2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix_tf = cv2.getPerspectiveTransform(ps1, ps2) 
    img_warp = cv2.warpPerspective(img, matrix_tf, (w, h))
    return img_warp

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

def transform(img):
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

def edge_detector(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
    return edges
    
def predict(img, clf):
    img_thres = thres(img)
    contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, max_area = biggestContour(contours)
    if (biggest.size != 0):
        # Detect contour
        edges = edge_detector(img)
        indices = np.where(edges != 0)
        x1, x2 = min(indices[0]), max(indices[0])
        y1, y2 = min(indices[1]), max(indices[1])
        p1, p2, p3, p4 = [y1, x1], [y1, x2], [y2,x1], [y2, x2]
        e = np.array([[p1],[p3],[p2],[p4]])
        img_warp = warp(img, e)
        img_filter = ndimage.median_filter(img_warp, 3)
        img_gray = cv2.cvtColor(img_filter, cv2.COLOR_RGB2GRAY)
        img_in = crop(img_gray, 5)
    else:
        # Do not detect contour
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_in = crop(img_gray, 6)
    img = transform(img_in)
    out = get_pred(clf(img.unsqueeze_(0)))
    return out[0]