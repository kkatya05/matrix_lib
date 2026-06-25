from __future__ import annotations

import numpy as np

from .base import Matrix
from .errors import MatrixNotSquareError, ShapeMismatchError, SingularMatrixError


def _plu_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	n = matrix.shape[0]
	p_matrix = np.eye(n, dtype=matrix.dtype)
	l_matrix = np.zeros((n, n), dtype=matrix.dtype)
	u_matrix = matrix.copy()

	for i in range(n):
		pivot_row = i + np.argmax(np.abs(u_matrix[i:, i]))
		pivot_value = u_matrix[pivot_row, i]
		if pivot_value == 0:
			raise SingularMatrixError("Linear system cannot be solved: matrix is singular")

		if pivot_row != i:
			u_matrix[[i, pivot_row]] = u_matrix[[pivot_row, i]]
			p_matrix[[i, pivot_row]] = p_matrix[[pivot_row, i]]
			l_matrix[[i, pivot_row], :i] = l_matrix[[pivot_row, i], :i]

		for j in range(i + 1, n):
			l_matrix[j, i] = u_matrix[j, i] / u_matrix[i, i]
			u_matrix[j, :] -= l_matrix[j, i] * u_matrix[i, :]

	for i in range(n):
		l_matrix[i, i] = 1

	return p_matrix, l_matrix, u_matrix


def _forward_substitution(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
	n = lower.shape[0]
	result = np.zeros_like(rhs, dtype=np.result_type(lower.dtype, rhs.dtype))

	if rhs.ndim == 1:
		for i in range(n):
			sum_value = 0
			for k in range(i):
				sum_value += lower[i, k] * result[k]
			result[i] = (rhs[i] - sum_value) / lower[i, i]
		return result

	for col in range(rhs.shape[1]):
		for i in range(n):
			sum_value = 0
			for k in range(i):
				sum_value += lower[i, k] * result[k, col]
			result[i, col] = (rhs[i, col] - sum_value) / lower[i, i]
	return result


def _backward_substitution(upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
	n = upper.shape[0]
	result = np.zeros_like(rhs, dtype=np.result_type(upper.dtype, rhs.dtype))

	if rhs.ndim == 1:
		for i in range(n - 1, -1, -1):
			sum_value = 0
			for k in range(n - 1, i, -1):
				sum_value += upper[i, k] * result[k]
			pivot = upper[i, i]
			if pivot == 0:
				raise SingularMatrixError("Linear system cannot be solved: matrix is singular")
			result[i] = (rhs[i] - sum_value) / pivot
		return result

	for col in range(rhs.shape[1]):
		for i in range(n - 1, -1, -1):
			sum_value = 0
			for k in range(n - 1, i, -1):
				sum_value += upper[i, k] * result[k, col]
			pivot = upper[i, i]
			if pivot == 0:
				raise SingularMatrixError("Linear system cannot be solved: matrix is singular")
			result[i, col] = (rhs[i, col] - sum_value) / pivot
	return result


def solve(A: Matrix, b: np.ndarray) -> np.ndarray:
	"""Решает систему линейных уравнений Ax = b через PLU-разложение."""

	if not isinstance(A, Matrix):
		raise TypeError("A must be a matrix from this library")

	if not isinstance(b, np.ndarray):
		raise TypeError("b must be a numpy.ndarray")

	if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
		raise MatrixNotSquareError(f"Matrix must be square, got shape {A.shape}")

	n = A.shape[0]
	if b.ndim == 1:
		if b.shape[0] != n:
			raise ShapeMismatchError(
				f"Right side length does not match matrix size: {b.shape[0]} != {n}"
			)
	elif b.ndim == 2:
		if b.shape[0] != n:
			raise ShapeMismatchError(
				f"Right side row count does not match matrix size: {b.shape[0]} != {n}"
			)
	else:
		raise ShapeMismatchError("Right side must be a vector or a 2-D array")

	p_matrix, l_matrix, u_matrix = _plu_decomposition(A.to_dense())
	rhs = p_matrix @ b
	y = _forward_substitution(l_matrix, rhs)
	return _backward_substitution(u_matrix, y)
