from .base import  Matrix
from .errors import ShapeMismatchError, SingularMatrixError
import numpy as np 

class FullMatrix(Matrix):
    """
    Заполненная матрица с элементами произвольного типа.
    """
    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data
        
    @property
    def shape(self):
        return self.data.shape
    
    def to_dense(self) -> np.ndarray:
        return self.data.copy()

    def matvec(self, x: np.ndarray) -> np.ndarray:
        # Проверка размеров (x должен подходить под количество столбцов)
        if len(x) != self.width:
            raise ShapeMismatchError(
            f"Vector length does not match matrix width"
        )
    
        # Умножение матрицы на вектор
        return self.data @ x
        
    
    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]       
        data = np.empty((height,width), dtype=dtype)
        return FullMatrix(data)
        
    @classmethod
    def zero(_cls, height, width, default=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return FullMatrix(data)
                    
    
    @property
    def dtype(self):
        return self.data.dtype
            
    def __getitem__(self, key):
        row, column = key
        return self.data[row, column]
    
    def __setitem__(self, key, value):
        row, column = key
        self.data[row, column] = value