from .base import  Matrix
from .errors import ShapeMismatchError, SingularMatrixError, MatrixNotSquareError
import numpy as np 

class SymmetricMatrix(Matrix):
    """Симметричная матрица, будем хранить только верхний треугольник"""
    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`
        """
        assert isinstance(data, np.ndarray)
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise MatrixNotSquareError(f"Symmetric matrix must be square, got shape {data.shape}")
        self.data = data

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        # Если размеры не указаны, берем текущий размер
        if width is None and height is None:
            size = self.data.shape[0]
        else:
            size = width if width is not None else height
            
        if width is not None and height is not None and width != height:
            raise MatrixNotSquareError("Symmetric matrix must be square (width == height)")
            
        data = np.empty((size, size), dtype=dtype)
        return SymmetricMatrix(data)

    @classmethod
    def empty(cls, size=1, dtype=float):
        data = np.empty((size, size), dtype=dtype)
        return cls(data)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __setitem__(self, key, value):
        row, column = key
        # Физически пишем только в верхний треугольник
        if column >= row:
            self.data[row, column] = value
        else:
            self.data[column, row] = value

    def __getitem__(self, key):
        row, column = key
        # Читаем нижний треугольник из верхнего
        if column < row:
            return self.data[column, row]
        return self.data[row, column]
    
    @classmethod
    def zero(cls, height, width=None, default=0, dtype=type(0)):
        """
        Создает матрицу размера `height` x `height` со значениями по умолчанию `default`
        """
        if width is not None and height != width:
            raise MatrixNotSquareError("Symmetric matrix must be square")
        data = np.empty((height, height), dtype=dtype)
        data[:] = default
        return cls(data)

    def to_dense(self) -> np.ndarray:
        """Конвертирует в обычный 2D numpy массив"""
        upper = np.triu(self.data)
        # Складываем верхний с нижним и убираем дублирование диагонали
        return upper + upper.T - np.diag(np.diag(upper))

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Умножение матрицы на вектор"""
        if len(x) != self.width:
            raise ShapeMismatchError("Vector length does not match matrix width")
        return self.to_dense() @ x

    def __sub__(self, other):
        from .dense import FullMatrix  # локальный импорт во избежание циклической зависимости
        
        if isinstance(other, SymmetricMatrix):
            if self.shape != other.shape:
                raise ShapeMismatchError("Matrix shapes must match for subtraction.")
            return SymmetricMatrix(self.data - other.data)
            
        if isinstance(other, FullMatrix):
            if self.shape != other.shape:
                raise ShapeMismatchError("Matrix shapes must match for subtraction.")
            # self.to_dense() возвращает чистый numpy-массив, из которого вычитаем other.data
            return FullMatrix(self.to_dense() - other.data)
            
        return NotImplemented

    def __matmul__(self, other):
        from .dense import FullMatrix
        
        if isinstance(other, FullMatrix):
            if self.width != other.height:
                raise ShapeMismatchError(f"Shapes do not match for matrix multiplication.")
            return FullMatrix(self.to_dense() @ other.data)
            
        if isinstance(other, Matrix):
            if self.width != other.height:
                raise ShapeMismatchError(f"Shapes do not match for matrix multiplication.")
            return FullMatrix(self.to_dense() @ other.to_dense())

        return NotImplemented