from cnn import get_pred
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import os
from tqdm import tqdm
from cnn import *
from knn import *

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
    img_crop = crop(img, 5)
    if (len(img_crop.shape) > 2):
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_crop
    img_invert = cv2.bitwise_not(img_gray)
    _, img_thres = cv2.threshold(img_invert, 127, 255, cv2.THRESH_BINARY)
    trans=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(28),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return trans(img_thres)

def preprocees_knn(img):
    img_crop = crop(img, 5)
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
    img_thres = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
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
    
def predict(img, clf, clf_type='knn'):
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
        img_in = ndimage.median_filter(img_warp, 3)
    else: img_in = img

    if clf_type == 'cnn':
        img_in = preprocess_cnn(img_in)
        out = get_pred(clf(img_in.unsqueeze_(0)))[0]
    
    elif clf_type == 'knn':
        img_in = preprocees_knn(img_in)
        out = int(clf.predict(img_in)[0])
    
    return out

def load_model(clf_type='knn'):
    if clf_type == 'cnn':
        model = CNN()
        device = torch.device('cpu')
        model.load_state_dict(torch.load(CNN_PATH, map_location=device))
        model.eval()
    elif clf_type == 'knn':
        model = pickle.load(open(KNN_PATH, 'rb'))
    return model