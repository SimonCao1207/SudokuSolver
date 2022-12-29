from utils import *
from sudokuSolver import *
import copy

PATH = 'model.pth'
# pathImage = "./img/sudoku_img1.png"
# pathImage = "./img/sudoku_img2.png"
pathImage = "./img/sudoku_img3.png"
heightImage = 450
widthImage = 450

# Prepare image
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImage, heightImage))
imgThreshold = preProcess(img)

#Find contours
imgContours, imgBigContour = img.copy(), img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 3)

biggest, max_area = biggestContour(contours)

if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImage, 0], [0, heightImage], [widthImage, heightImage]])
    matrix_tf = cv2.getPerspectiveTransform(pts1, pts2) 
    imgWarpColored = cv2.warpPerspective(img, matrix_tf, (widthImage, heightImage))
    imgWarpColored = ndimage.median_filter(imgWarpColored, 3)
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_RGB2GRAY)
    boxes = split_boxes(imgWarpColored) 
    
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
    
    # import os
    # base = "./img/sudoku_label3"
    # for i, box in enumerate(boxes):
    #     path = os.path.join(base, str(label_3[i]))
    #     print(f"Getting {i} :  {path}")
    #     getImage(path, i, box)
    # exit()

    predictions = []
    for i, box in enumerate(tqdm(boxes)):
        copy_box = copy.deepcopy(box)
        if isWhite(copy_box):
            predictions.append(0)
            continue
            
        # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, blackAndWhiteImage = cv2.threshold(np.array(img), 127, 255, cv2.THRESH_BINARY)
        # im_pil = Image.fromarray(blackAndWhiteImage)
        # imtest = transforms.ToTensor()(im_pil).unsqueeze_(0)

        img = preprocess(box)
        model = load_model()
        num = get_pred(model(img.unsqueeze_(0)))
        predictions.append(num.detach().numpy()[0])
        grid = []
        for i in range(0, 73, 9):
            grid.append(predictions[i:i+9])


print_grid(grid)
# if solve_sudoku(grid):
#     print_grid(grid)
# else: print("This puzzle is not solvable")



