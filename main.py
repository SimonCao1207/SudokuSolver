from utils import *
from sudokuSolver import *

PATH = 'model.pth'
# pathImage = "./img/sudoku_img1.PNG"
pathImage = "./img/sudoku_img2.png"
heightImage = 450
widthImage = 450

# Prepare image
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImage, heightImage))
imgThreshold = preProcess(img)

# exit(0)

#Find contours
imgContours, imgBigContour = img.copy(), img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 3)

# display(imgContours)
# exit(0)

biggest, max_area = biggestContour(contours)

if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImage, 0], [0, heightImage], [widthImage, heightImage]])
    matrix_tf = cv2.getPerspectiveTransform(pts1, pts2) 
    imgWarpColored = cv2.warpPerspective(img, matrix_tf, (widthImage, heightImage))
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_RGB2GRAY)

    # SPLIT the image into boxes of number
    boxes = split_boxes(imgWarpColored)
    numbers = predict(boxes)

display(imgWarpColored)
grid = []
for i in range(9):
    grid.append(numbers[i:i+9])

if solve_sudoku(grid):
    for c in grid:
        print(c)
else: print("This puzzle is not solvable")



