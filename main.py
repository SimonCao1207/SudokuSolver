from utils import *
from sudokuSolver import *
import copy

# pathImage = "./img/sudoku_img1.png"
# pathImage = "./img/sudoku_img2.png"
pathImage = "./img/sudoku_img3.png"
h, w = 450, 450

# Prepare image
img = cv2.imread(pathImage)
img = cv2.resize(img, (w, h))

#Find contours
# imgContours, imgCorners = img.copy(), img.copy()
img_thres = thres(img)
contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 3)
biggest, max_area = biggestContour(contours)

if biggest.size != 0:
    biggest = reorder(biggest)
    # cv2.drawContours(imgCorners, biggest, -1, (0, 0, 255), 10)
    img_warp = warp(img, biggest)
    img_filter = ndimage.median_filter(img_warp, 3)
    img_gray = cv2.cvtColor(img_filter, cv2.COLOR_RGB2GRAY)
    boxes = split_boxes(img_gray) 
    
    # label_3 = [
    #    0, 0, 8, 0, 0, 0, 0, 0, 0, 
    #    4, 9, 0, 1, 5, 7, 0, 0, 2, 
    #    0, 0, 3, 0, 0, 4, 1, 9, 0,
    #    1, 8, 5, 0, 6, 0, 0, 2, 0, 
    #    0, 0, 0, 0, 2, 0, 0, 6, 0, 
    #    9, 6, 0, 4, 0, 0, 3, 0, 0, 
    #    0, 0, 0, 0, 7, 2, 0, 0, 4, 
    #    0, 4, 9, 0, 3, 0, 0, 5, 7, 
    #    8, 2, 7, 0, 0, 9, 0, 1, 3
    # ]
    
    # label_2 = [
    #    3, 4, 2, 7, 0, 0, 0, 0, 0, 
    #    0, 0, 0, 0, 2, 4, 0, 0, 1, 
    #    0, 9, 0, 5, 0, 0, 3, 0, 0,
    #    0, 0, 0, 0, 0, 2, 0, 0, 6, 
    #    4, 0, 0, 0, 0, 0, 0, 8, 2, 
    #    0, 0, 0, 0, 0, 9, 7, 0, 0, 
    #    0, 0, 0, 0, 7, 0, 0, 0, 0, 
    #    0, 5, 0, 4, 0, 0, 6, 0, 0, 
    #    0, 0, 6, 0, 0, 0, 0, 0, 0
    # ]

    # label_1 = [
    #    0, 0, 6, 7, 0, 3, 2, 0, 0, 
    #    0, 5, 8, 4, 0, 0, 0, 6, 0, 
    #    4, 0, 0, 0, 9, 0, 0, 7, 1,
    #    8, 0, 0, 0, 3, 0, 0, 5, 4, 
    #    0, 0, 2, 9, 0, 7, 6, 0, 0, 
    #    6, 3, 0, 0, 4, 0, 0, 0, 9, 
    #    1, 8, 0, 0, 6, 0, 0, 0, 7, 
    #    0, 6, 0, 0, 0, 9, 4, 8, 0, 
    #    0, 0, 5, 8, 0, 4, 3, 0, 0
    # ]

    # import os
    # base = "./img/sudoku_label2"
    # for i, box in enumerate(boxes):
    #     path = os.path.join(base, str(label_2[i]))
    #     print(f"Writing {i} :  {path}")
    #     get_image(path, i, box)
    # exit()

    clf = load_model()
    predictions = []
    for i, box in enumerate(tqdm(boxes)):
        copy_box = copy.deepcopy(box)
        if isWhite(copy_box):
            predictions.append(0)
        else: 
            img = transform(box)
            num = get_pred(clf(img.unsqueeze_(0)))
            predictions.append(num.detach().numpy()[0])
            grid = []
    for i in range(0, 73, 9):
        grid.append(predictions[i:i+9])


print_grid(grid)
# if solve_sudoku(grid):
#     print_grid(grid)
# else: print("This puzzle is not solvable")



