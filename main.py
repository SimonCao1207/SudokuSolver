from utils import *
from solver import *
from scipy import ndimage
from PIL import Image

if __name__ == "__main__":
    # pathImage = "./img/sample_img1.png"
    # pathImage = "./img/sample_img4.png"
    pathImage = "./img/sample_img5.png"

    H, W = 450, 450
    print("Reading the image: ")
    img = cv2.imread(pathImage)
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
        clf = load_model("svhn")
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
    print(f"Sudoku board read from image: {pathImage}")
    print_grid(grid)
    print("-"*20)
    if solve_sudoku(grid):
        print("Solved !")
        print_grid(grid)
    else: print("This puzzle is not solvable")



