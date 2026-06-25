# tests/test_solvers.py
import pytest
import numpy as np
from src import FullMatrix, solve, MatrixNotSquareError, ShapeMismatchError, SingularMatrixError

def test_solve_vector_rhs():
    """ Обычная хорошо обусловленная система Ax = b """
    A = FullMatrix(np.array([[2.0, 1.0], [1.0, 3.0]]))
    b = np.array([11.0, 13.0])
    
    x = solve(A, b)
    expected = np.array([4.0, 3.0])
    np.testing.assert_allclose(x, expected, rtol=1e-9)

def test_solve_matrix_rhs():
    """ Правая часть в виде матрицы (несколько векторов b) """
    A = FullMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    X = solve(A, B)
    expected = np.linalg.solve(A.to_dense(), B)
    np.testing.assert_allclose(X, expected, rtol=1e-9)

def test_solve_non_square_matrix():
    """ Размерность b меньше A """
    A = FullMatrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    b = np.array([1.0, 2.0])
    
    with pytest.raises(MatrixNotSquareError):
        solve(A, b)

def test_solve_dimension_mismatch():
    """ Размерность b больше A """
    A = FullMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ShapeMismatchError):
        solve(A, b)

def test_solve_singular_matrix():
    """ Вырожденная матрица (вторая строка линейно зависима) """
    A = FullMatrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    b = np.array([5.0, 10.0])
    
    with pytest.raises(SingularMatrixError):
        solve(A, b)