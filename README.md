## Sudoku Solver


Input image            |  Solved output
:-------------------------:|:-------------------------:
<img src="https://github.com/SimonCao1207/SudokuSolver/blob/master/img/sample_img2.png?raw=True" alt="drawing" width="300"/> |  <img src="https://github.com/SimonCao1207/SudokuSolver/blob/master/img/solved.png?raw=True" alt="drawing" width="300"/>






## Getting started
  - Train a digit classification model from [SVHN](http://ufldl.stanford.edu/housenumbers/) and MNIST dataset to recognize number from user's input image. Checkpoints are saved in `./checkpoints`

  - You can try train the model again in `./svhn`

  - Run `main.py` to see the demo 
    ```sh
    pip install -r requirements.txt
    python main.py
    ```

## Run test

  ```sh
  python test/test_utils.py
  ```


### TODO: 

- [ ] Improve digit classifier, currently is not working properly for certain images (maybe because of preprocessing phase)

- [ ] Front end ? 



