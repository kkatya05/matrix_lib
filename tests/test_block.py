# tests/test_block.py
import pytest
import numpy as np
from src import FullMatrix, ShapeMismatchError
from src.block import BlockMatrix
@pytest.fixture
def sample_blocks():
    """ Инициализация блоков """
    b11 = FullMatrix(np.array([[1, 2], [3, 4]]))
    b12 = FullMatrix(np.array([[5], [6]]))
    b21 = FullMatrix(np.array([[7, 8]]))
    b22 = FullMatrix(np.array([[9]]))
    return [[b11, b12], [b21, b22]]

def test_block_matrix_creation_and_properties(sample_blocks):
    """ Проверка формирования матрицы """
    block_mat = BlockMatrix(sample_blocks)
    assert block_mat.shape == (3, 3)
    assert block_mat.block_row_heights == [2, 1]
    assert block_mat.block_col_widths == [2, 1]

def test_block_matrix_validation():
    """ Проверка на формирование разных размерностей"""
    b1 = FullMatrix(np.array([[1, 2]]))
    b2 = FullMatrix(np.array([[3, 4, 5]]))
    
    with pytest.raises(ShapeMismatchError):
        BlockMatrix([[b1, b2], [b1]])
        
    b_wrong_height = FullMatrix(np.array([[1], [2]]))
    with pytest.raises(ShapeMismatchError):
        BlockMatrix([[b1, b_wrong_height]])

def test_to_dense(sample_blocks):
    """ Правильно ли сформировалась итоговая матрица"""
    block_mat = BlockMatrix(sample_blocks)
    expected = np.array([
        [1, 2, 5],
        [3, 4, 6],
        [7, 8, 9]
    ])
    np.testing.assert_array_equal(block_mat.to_dense(), expected)

def test_block_indexing(sample_blocks):
    """ Проверка на чтение и запись элементов """
    block_mat = BlockMatrix(sample_blocks)
    
    assert block_mat[0, 0] == 1
    assert block_mat[1, 2] == 6
    assert block_mat[2, 0] == 7
    assert block_mat[2, 2] == 9
    
    block_mat[1, 2] = 99
    assert block_mat[1, 2] == 99
    assert sample_blocks[0][1][1, 0] == 99 

    with pytest.raises(IndexError):
        _ = block_mat[3, 3]

def test_block_matvec(sample_blocks):
    """ Проверка арифметических операций """
    block_mat = BlockMatrix(sample_blocks)
    x = np.array([1, 2, 3])
    res = block_mat.matvec(x)
    
    expected = block_mat.to_dense() @ x
    np.testing.assert_array_equal(res, expected)