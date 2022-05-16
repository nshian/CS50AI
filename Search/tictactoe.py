"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0
    for row in board:
        x_count += row.count(X)
        o_count += row.count(O)

    if x_count > o_count:
        return O
    return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for row_idx, row in enumerate(board):
        for col_idx, val in enumerate(row):
            if val == EMPTY:
                actions.add((row_idx, col_idx))
    return actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    row, col = action
    temp = deepcopy(board)
    person = player(board)
    if board[row][col] != EMPTY:
        raise Exception
    else:
        temp[row][col] = person

    return temp

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for person in (X, O):
        for row in board: #checking rows
            if row.count(person) == 3:
                return person

        for i in range(3): #checking cols
            count = 0
            for j in range(3):
                if board[j][i] == person:
                    count += 1
            if count == 3:
                return person

        if board[1][1] == person:
            if (board[0][0] == person and board[2][2] == person):
                return person
            elif (board[2][0] == person and board [0][2] == person):
                return person
            return None

        return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:
        return True

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                return False
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def max_val(board):
        optimal = ()
        v = -100
        for action in actions(board):
            min = min_val(result(board, action))[0]
            if min > v:
                v = min
                optimal = action
        return v, optimal

    def min_val(board):
        optimal = ()
        v = 100
        for action in actions(board):
            max = max_val(result(board, action))[0]
            if max < v:
                v = max
                optimal = action
        return v, optimal

    if terminal(board):
        return None

    person = player(board)
    optimal = ()
    if person == X:
        optimal = max_val(board)[1]
    else:
        optimal = min_val(board)[1]
    return optimal