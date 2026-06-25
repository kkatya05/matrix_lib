# tests/test_dense.py
import pytest
import numpy as np
from src import FullMatrix, ShapeMismatchError

def test_full_matrix_creation_and_properties():
    """ Проверка создания матриц и проверка свойств """
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix = FullMatrix(data)
    
    assert matrix.shape == (2, 2)
    assert matrix.width == 2
    assert matrix.height == 2
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix.to_dense(), data)

def test_getitem_setitem():
    """ Проверка получения элементов """
    data = np.array([[10, 20], [30, 40]], dtype=int)
    matrix = FullMatrix(data)
    
    assert matrix[0, 1] == 20
    matrix[1, 0] = 99
    assert matrix[1, 0] == 99
    assert matrix.data[1, 0] == 99

def test_empty_like():
    """ Создание пустой матрицы соответствующей размерности """
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix = FullMatrix(data)
    
    empty_mat = matrix.empty_like()
    assert empty_mat.shape == matrix.shape
    assert empty_mat.dtype == matrix.dtype
    
    custom_empty = matrix.empty_like(width=5, height=2)
    assert custom_empty.shape == (2, 5)

def test_zero_classmethod():
    """ Создание нулевой матрицы """
    matrix = FullMatrix.zero(height=3, width=4, default=5.0)
    assert matrix.shape == (3, 4)
    assert matrix.dtype == np.float64
    expected = np.full((3, 4), 5.0)
    np.testing.assert_array_equal(matrix.to_dense(), expected)

def test_matvec_product():
    """ Проверка операций матрица & вектор """
    data = np.array([[1, 2], [3, 4]])
    matrix = FullMatrix(data)
    vec = np.array([10, 20])
    
    res = matrix.matvec(vec)
    np.testing.assert_array_equal(res, np.array([50, 110]))
    
    wrong_vec = np.array([1, 2, 3])
    with pytest.raises(ShapeMismatchError):
        matrix.matvec(wrong_vec)

def test_arithmetic_operations():
    """ Проверка операций матрица & матрица"""
    a = FullMatrix(np.array([[1, 2], [3, 4]]))
    b = FullMatrix(np.array([[5, 6], [7, 8]]))
    
    diff = a - b
    np.testing.assert_array_equal(diff.to_dense(), np.array([[-4, -4], [-4, -4]]))

    prod = a @ b
    np.testing.assert_array_equal(prod.to_dense(), np.array([[19, 22], [43, 50]]))