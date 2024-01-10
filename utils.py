import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import os
from tqdm import tqdm
import torch
from cnn import *
import pickle


KNN_PATH = "./checkpoints/knn.sav"

def display(img, name='board'): 
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def thres(img):
    """
    Adaptive Thresholding
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
    img_thres = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_thres

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
    ps: array of 4 points
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

def preprocess_cnn(img):
    
    img = PIL.ImageOps.invert(img)

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(40),
            transforms.Resize(28, antialias=True),
    ])
    return transform(img)

def preprocees_knn(img):
    img_crop = crop(img, 4)
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY) if (len(img_crop.shape) > 2) else img_crop
    img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
    img_thres = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    # img_thres = zero_pad(img_thres, 5)
    img_invert = cv2.bitwise_not(img_thres)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    img_process = cv2.dilate(img_invert, kernel)
    img_process = cv2.resize(img_process, (28,28), interpolation=cv2.INTER_AREA)
    img_in = img_process.reshape(1, -1)
    return img_in

def print_grid(grid):
    for r in grid:
        print(r)
        
def get_image(path, num, box):
    cv2.imwrite(os.path.join(path, f"{num}.png"), box)

def crop(img, c):
    if (len(img.shape) > 2): 
        h, w, _ = img.shape
        img_crop = img[c:h-c, c:w-c, :]
    else:
        h, w = img.shape
        img_crop = img[c:h-c, c:w-c]
    return img_crop

def whiten_edge(img, c):
    if (len(img.shape) > 2):
        h, w, _ = img.shape
    else: h, w = img.shape
    img_white = np.ones_like(img)*255
    img_white[c:h-c, c:w-c] = img[c:h-c, c:w-c]
    return img_white

def isWhite(img, thres=40):
    img_process = whiten_edge(img, 10)
    _, img_thres = cv2.threshold(img_process, 127, 255, cv2.THRESH_BINARY)
    img_invert = cv2.bitwise_not(img_thres)
    num_black_pix = np.count_nonzero(img_invert)
    if num_black_pix > thres:
        return False
    return True

def edge_detector(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
    return edges
    
def predict(img_in, clf, clf_type='knn'):
    
    if clf_type == 'cnn':
        img_in = preprocess_cnn(img_in)
        out = get_pred(clf(img_in.unsqueeze(0)))
    
    elif clf_type == 'knn':
        img_in = preprocees_knn(img_in)
        out = int(clf.predict(img_in)[0])
    
    return out