# tests/test_integration.py
import pytest
import numpy as np
from src import FullMatrix, SymmetricMatrix, solve
from src.block import BlockMatrix

def test_symmetric_and_dense_interaction():
    """ Проверка """
    # Создаем симметричную матрицу
    sym_data = np.array([[4.0, 1.0], [1.0, 3.0]])
    A_sym = SymmetricMatrix(sym_data)
    
    # Создаем плотную матрицу
    dense_data = np.array([[1.0, 0.0], [2.0, 5.0]])
    B_dense = FullMatrix(dense_data)
    
    # 1. Складываем/вычитаем их через общий интерфейс
    diff = A_sym - B_dense
    expected_diff = A_sym.to_dense() - B_dense.to_dense()
    np.testing.assert_array_equal(diff.to_dense(), expected_diff)
    
    # 2. Перемножаем их матрицы
    prod = A_sym @ B_dense
    expected_prod = A_sym.to_dense() @ B_dense.to_dense()
    np.testing.assert_array_equal(prod.to_dense(), expected_prod)

def test_mixed_block_matrix():
    # Собираем BlockMatrix, где один из блоков — SymmetricMatrix
    sym_block = SymmetricMatrix(np.array([[2.0, 0.0], [0.0, 2.0]]))
    dense_block = FullMatrix(np.array([[1.0], [3.0]]))
    
    row_blocks = [[sym_block, dense_block]]
    block_mat = BlockMatrix(row_blocks)
    
    assert block_mat.shape == (2, 3)
    expected_dense = np.array([
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 3.0]
    ])
    np.testing.assert_array_equal(block_mat.to_dense(), expected_dense)

def test_solve_system_with_symmetric_matrix():
    # Проверяем, может ли решатель работать с объектом SymmetricMatrix
    A = SymmetricMatrix(np.array([[4.0, 2.0], [2.0, 5.0]]))
    b = np.array([10.0, 11.0])
    
    x = solve(A, b)
    expected = np.linalg.solve(A.to_dense(), b)
    np.testing.assert_allclose(x, expected, rtol=1e-9)

def test_matrix_repr_string_formatting():
    # Интеграционный тест красивого вывода __repr__, унаследованного от Matrix
    # Проверяем на FullMatrix
    A = FullMatrix(np.array([[1, 12], [123, 4]]))
    repr_str = repr(A)
    
    # Проверяем базовые маркеры форматирования границ, заложенные в base.py
    assert "|" in repr_str
    # Проверяем, что элементы попали в строковое представление
    assert "123" in repr_str
    assert "12" in repr_str