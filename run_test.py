import os
import warnings
from utils import isWhite, load_model, predict
import cv2
from termcolor import colored
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--t', default='0')
args = parser.parse_args()

warnings.filterwarnings("ignore")

PASS = colored("[PASS]", 'green')
FAIL = colored('[FAIL]','red')

dct = {
    '0' : 'clf',
    '1' : 'isWhite',
}

def _print_final(num_pass, num_fail=0, _pass=True):
    if _pass:
        print(f"{PASS} all {num_pass} tests")
    else:
        print(f"--> {FAIL}: {colored(f'{num_fail}', 'red')} / {colored(f'{num_fail + num_pass}', 'green')} tests")


def test_isWhite(base = "./img/sudoku_label3"):
    N = len(os.listdir(base))
    cnt = 0
    for i in range(N):
        base_path = os.path.join(base, str(i))
        dirs = os.listdir(base_path)
        for file in dirs:
            path = os.path.join(base_path, file)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if isWhite(img):
                if (i==0):
                    cnt+=1
                    print(PASS)
                else: print(f"{FAIL}: {path}")
            else:
                if (i==0):
                    print(f"{FAIL}: {path}")
                else: 
                    cnt+=1
                    print(PASS)
    if (cnt == 81):
        _print_final(81)
    else: 
        _print_final(cnt, 81-cnt, False)

def test_clf(base="./img/sudoku_label3", clf_type='knn'):
    N = len(os.listdir(base))
    num_files = 81 - len(os.listdir(os.path.join(base, '0')))
    cnt = 0
    clf = load_model(clf_type=clf_type)
    for i in range(1, N):
        base_path = os.path.join(base, str(i))
        dirs = os.listdir(base_path)
        for file in dirs:
            path = os.path.join(base_path, file)
            img = cv2.imread(path)
            pred = predict(img, clf=clf, clf_type=clf_type)
            ####### 
            # if (pred != i):
            #     clf_cnn = load_model(clf_type='cnn')
            #     pred = predict(img, clf_cnn, clf_type="cnn") 
            ######

            if (pred == i):
                cnt += 1
                print(PASS)
            else: print(f"{FAIL}: {path}")
    if (cnt == num_files):
        _print_final(num_files)
    else: 
        _print_final(cnt, num_files-cnt, False)

    
if __name__ == "__main__":
    name = dct[args.t]
    if (name == 'clf'):
        test_clf(base="./img/sudoku_label1", clf_type='knn')
        # test_clf(base="./img/sudoku_label1", clf_type='cnn')
        # test_clf(base="./img/sudoku_label2", clf_type='knn')
        # test_clf(base="./img/sudoku_label2", clf_type='cnn')
        # test_clf(base="./img/sudoku_label3", clf_type='knn')
        # test_clf(base="./img/sudoku_label3", clf_type='cnn')
    if (name == 'isWhite'):
        # test_isWhite(base="./img/sudoku_label3")
        # test_isWhite(base="./img/sudoku_label2")
        test_isWhite(base="./img/sudoku_label1")