import os
import warnings
from utils import *
from termcolor import colored
warnings.filterwarnings("ignore")

PASS = colored("[PASS]", 'green')
FAIL = colored('[FAIL]','red')

def test_isWhite():
    base = "./img/sudoku_label3"
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
        print(f"{PASS} all 81 tests")
    else: 
        print(f"{FAIL}: {colored(f'{81 - cnt}', 'red')} / {colored('81', 'green')} tests")
    
if __name__ == "__main__":
    test_isWhite()