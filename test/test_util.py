import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from PIL import Image
from utils import *
from solver import *
from scipy import ndimage
from delayed_assert import delayed_assert, expect, assert_expectations

# LABEL_PATH = "./img/sudoku_label3"
# LABEL_PATH = "./img/sudoku_label2"
LABEL_PATH = "./img/sudoku_label1"

sol1 = [
    [9, 1, 6, 7, 8, 3, 2, 4, 5],                                                                                                                                                                   
    [7, 5, 8, 4, 2, 1, 9, 6, 3],                                                                                                                                                                   
    [4, 2, 3, 6, 9, 5, 8, 7, 1],                                                                                                                                                                   
    [8, 7, 9, 2, 3, 6, 1, 5, 4],                                                                                                                                                                   
    [5, 4, 2, 9, 1, 7, 6, 3, 8],                                                                                                                                                                   
    [6, 3, 1, 5, 4, 8, 7, 2, 9],                                                                                                                                                                   
    [1, 8, 4, 3, 6, 2, 5, 9, 7],                                                                                                                                                                   
    [3, 6, 7, 1, 5, 9, 4, 8, 2],                                                                                                                                                                   
    [2, 9, 5, 8, 7, 4, 3, 1, 6],
]

sol2 = [
    [3, 4, 2, 7, 9, 1, 8, 6, 5],
    [8, 6, 5, 3, 2, 4, 9, 7, 1],
    [7, 9, 1, 5, 6, 8, 3, 2, 4],
    [5, 7, 3, 8, 4, 2, 1, 9, 6],
    [4, 1, 9, 6, 3, 7, 5, 8, 2],
    [6, 2, 8, 1, 5, 9, 7, 4, 3],
    [1, 3, 4, 9, 7, 6, 2, 5, 8],
    [2, 5, 7, 4, 8, 3, 6, 1, 9],
    [9, 8, 6, 2, 1, 5, 4, 3, 7]
]

test_suits = [
    {   
        "pathImage" : "./img/sudoku_img1.png",
        "sol" : sol1
    },
    {   
        "pathImage" : "./img/sudoku_img2.png",
        "sol" : sol2
    },
]

class TestUtils(unittest.TestCase):

    def test_isWhite_pos(self):
        N = len(os.listdir(LABEL_PATH))
        num_zero = 0
        pred_zero = 0
        for i in range(N):
            base_path = os.path.join(LABEL_PATH, str(i))
            dirs = os.listdir(base_path)
            for file in dirs:
                path = os.path.join(base_path, file)
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if isWhite(img):
                    pred_zero += 1
                if i == 0:  
                    num_zero += 1
        self.assertEqual(pred_zero, num_zero)

    def test_clf(self):
        N = len(os.listdir(LABEL_PATH))
        clf = load_model()
        for i in range(1, N):
            base_path = os.path.join(LABEL_PATH, str(i))
            dirs = os.listdir(base_path)
            for file in dirs:
                row, col = int(int(file.split(".")[0]) / 9), int(file.split(".")[0]) % 9 
                path = os.path.join(base_path, file)
                img = Image.open(path)
                pred = predict(img, clf=clf)
                expect(lambda: self.assertEqual(pred, i, f"at row = {row}, col = {col}"))    
        assert_expectations()

    def _helper_end2end(self, test_num=0):
        H, W = 450, 450
        test = test_suits[test_num]
        img = cv2.imread(test["pathImage"])
        img = cv2.resize(img, (W, H))

        img_thres = thres(img)
        contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest, max_area = biggestContour(contours)

        if biggest.size != 0:
            biggest = reorder(biggest)
            img_warp = warp(img, biggest)
            img_filter = ndimage.median_filter(img_warp, 3)
            img_gray = cv2.cvtColor(img_filter, cv2.COLOR_RGB2GRAY)
            boxes = split_boxes(img_gray) 
            clf = load_model()
            predictions = []
            for i, box in enumerate(boxes):
                if isWhite(box):
                    predictions.append(0)
                else:
                    im_pil = Image.fromarray(box) # convert to pil image
                    pred = predict(im_pil, clf=clf)
                    predictions.append(pred)
                    grid = []
            for i in range(0, 73, 9):
                grid.append(predictions[i:i+9])
        return solve_sudoku(grid), grid
    
    def test_end2end_1(self):
        is_solve, grid = self._helper_end2end(0)
        self.assertEqual(is_solve, True)
        self.assertEqual(grid, test_suits[0]["sol"])
    
    def test_end2end_2(self):
        is_solve, grid = self._helper_end2end(1)
        self.assertEqual(is_solve, True)
        self.assertEqual(grid, test_suits[1]["sol"])
            
if __name__ == "__main__":
    unittest.main()
