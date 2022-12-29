
def find_next_empty(puzzle):
    """
    return tuple (row, col) if there is an empty cell, otherwise (None, None) 
    """
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                return r,c
    return None, None

def is_valid(puzzle, guess, row, col):
    """
    return True if the guess is valid, otherwise False
    """
    row_vals = puzzle[row]
    if guess in row_vals:
        return False
    col_vals = [puzzle[i][col] for i in range(9)]
    if guess in col_vals:
        return False
    row_start = (row//3)*3
    col_start = (col//3)*3
    for r in range(row_start, row_start+3):
        for c in range(col_start, col_start+3):
            if guess == puzzle[r][c]:
                return False
    return True


def solve_sudoku(puzzle):
    """
    input: 2-D array of sudoku puzzle board with 0 being an empty cell.
    return: True if the puzzle is solvable, otherwise False
    """
    row, col = find_next_empty(puzzle)
    if row is None:
        return True
    for guess in range(1,10):
        if is_valid(puzzle, guess, row, col):
            puzzle[row][col] = guess
            if solve_sudoku(puzzle):
                return True
        puzzle[row][col] = 0
    return False

if __name__ == '__main__':
    example_board = [
       [0, 0, 8, 0, 0, 0, 0, 0, 0], 
       [4, 9, 0, 1, 5, 7, 0, 0, 2], 
       [0, 0, 3, 0, 0, 4, 1, 9, 0],
       [1, 8, 5, 0, 6, 0, 0, 2, 0], 
       [0, 0, 0, 0, 2, 0, 0, 6, 0], 
       [9, 6, 0, 4, 0, 0, 3, 0, 0], 
       [0, 0, 0, 0, 7, 2, 0, 0, 4], 
       [0, 4, 9, 0, 3, 0, 0, 5, 7], 
       [8, 2, 7, 0, 0, 9, 0, 1, 3]
    ]
    if solve_sudoku(example_board):
        for c in example_board:
            print(c)
    else: print("This puzzle is not solvable")