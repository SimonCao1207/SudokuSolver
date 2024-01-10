from utils import *
from solver import *
from scipy import ndimage
from PIL import Image

if __name__ == "__main__":
    pathImage = "./img/sudoku_img1.png"
    H, W = 450, 450
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
        clf = load_model()
        predictions = []
        for i, box in enumerate(tqdm(boxes)):
            if isWhite(box):
                predictions.append(0)
            else:
                im_pil = Image.fromarray(box) # convert to pil image
                pred = predict(im_pil, clf=clf, clf_type="cnn")
                predictions.append(pred)
                grid = []
        for i in range(0, 73, 9):
            grid.append(predictions[i:i+9])

    if solve_sudoku(grid):
        print_grid(grid)
    else: print("This puzzle is not solvable")



