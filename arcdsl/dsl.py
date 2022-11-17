import numpy as np
from typing import Protocol


class Transform(Protocol):
    def __call__(self, input_matrix: np.array) -> np.array:
        pass


class ConstMatrix:
    """
    Transforms the input matrix (or a submatrix) into the given const matrix.
    """

    def __init__(self, const_matrix: np.array):
        self.const_matrix = const_matrix

    def __call__(self, _) -> np.array:
        return self.const_matrix


class Conditional:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, input_matrix: np.array) -> np.array:
        for cond, transform in self.mapping:
            if cond(input_matrix):
                return transform(input_matrix)
        raise ValueError("no condition met")
    

class EachSubmatrix:
    """
    Applies a transformation to each submatrix of the given submatrix_shape and then assembles the results.
    """

    def __init__(self, submatrix_shape: tuple, transform: Transform):
        self.submatrix_shape = submatrix_shape
        self.transform = transform

    def __call__(self, input_matrix: np.array) -> np.array:
        res = []
        for row in range(0, input_matrix.shape[0], self.submatrix_shape[0]):
            res_row = []
            for col in range(0, input_matrix.shape[1], self.submatrix_shape[1]):
                submatrix = input_matrix[row:row + self.submatrix_shape[0], col:col + self.submatrix_shape[1]]
                res_row.append(self.transform(submatrix))
            res.append(np.concatenate(res_row, axis=1))
        return np.concatenate(res, axis=0)


class RepeatHorizontalPattern:
    """
    Identifies a horizontal pattern and extends the input matrix by the given amount of additional columns. 
    """

    def __init__(self, additional_columns: int):
        self.additional_columns = additional_columns

    def __call__(self, input_matrix: np.array) -> np.array:
        columns = []
        pattern_discovered = False
        for column_i in range(input_matrix.shape[1]):
            column = input_matrix[:, column_i]
            if pattern_discovered:
                assert np.array_equal(column, columns[0])
            if len(columns) < 1 or not np.array_equal(columns[0], column):
                columns.append(column)
            else:
                pattern_discovered = True
                columns = columns[1:] + columns[:1]
        added_columns = []
        for _ in range(self.additional_columns):
            added_columns.append(columns[0])
            columns = columns[1:] + columns[:1]
        return np.hstack([input_matrix, np.array(added_columns).T])


def create_foreach_cell(transform: Transform) -> Transform:
    return EachSubmatrix((1, 1), transform)


def create_foreach_row(transform: Transform, row_size: int) -> Transform:
    return EachSubmatrix((1, row_size), transform)


def create_foreach_col(transform: Transform, col_size: int) -> Transform:
    return EachSubmatrix((col_size, 1), transform)


class MaskByValue:
    """
    Returns the mask of the input_matrix where the given value is found
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self, input_matrix: np.array) -> np.array:
        return (input_matrix == self.value).astype(np.uint8)


class ReplaceByMask:
    """
    Returns a copy of the input_matrix, with cells that match the mask being replaced by the given value.

    >>> ReplaceByMask(np.array([[0, 1], [0, 0]]), 7)(np.array([[3, 3], [3, 3]]))
    array([[3, 7],
           [3, 3]])
    """

    def __init__(self, mask: np.array, value: int) -> None:
        self.mask = mask
        self.value = value

    def __call__(self, input_matrix: np.array) -> np.array:
        return np.where(self.mask, self.value, input_matrix)


if __name__ == "__main__":
    import doctest
    doctest.testmod()