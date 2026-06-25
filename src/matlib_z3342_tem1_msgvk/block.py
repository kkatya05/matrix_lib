from .base import  Matrix
from .dense import  FullMatrix
from .errors import ShapeMismatchError, SingularMatrixError, MatrixNotSquareError
import numpy as np 

class BlockMatrix(Matrix):
    """Блочная матрица, составленная из прямоугольной таблицы блоков."""
    
    def __init__(self, blocks):
        """
        Создает блочную матрицу из списка списков блоков.
        
        Args:
            blocks: список списков блоков (каждый блок - Matrix или приводимый к FullMatrix)
        """
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise ShapeMismatchError("Blocks must be a non-empty list")
        
        # Преобразуем все блоки в объекты Matrix
        self.blocks = []
        for row_idx, row in enumerate(blocks):
            if not isinstance(row, list):
                raise ShapeMismatchError(f"Row {row_idx} must be a list")
            
            block_row = []
            for col_idx, block in enumerate(row):
                if not isinstance(block, Matrix):
                    # Пытаемся преобразовать к FullMatrix
                    block = FullMatrix(block)
                block_row.append(block)
            self.blocks.append(block_row)
        
        # Проверяем, что все строки имеют одинаковую длину (прямоугольная таблица)
        num_cols = len(self.blocks[0])
        for row_idx, row in enumerate(self.blocks):
            if len(row) != num_cols:
                raise ShapeMismatchError(
                    f"Row {row_idx} has {len(row)} blocks, but expected {num_cols}"
                )
        
        # Проверяем совместимость высот блоков в каждой строке
        self.block_row_heights = []
        for row_idx, row in enumerate(self.blocks):
            if len(row) == 0:
                raise ShapeMismatchError(f"Row {row_idx} is empty")
            height = row[0].height
            for col_idx, block in enumerate(row):
                if block.height != height:
                    raise ShapeMismatchError(
                        f"Block at ({row_idx}, {col_idx}) has height {block.height}, "
                        f"but expected {height}"
                    )
            self.block_row_heights.append(height)
        
        # Проверяем совместимость ширин блоков в каждом столбце
        self.block_col_widths = []
        for col_idx in range(num_cols):
            width = self.blocks[0][col_idx].width
            for row_idx in range(len(self.blocks)):
                block = self.blocks[row_idx][col_idx]
                if block.width != width:
                    raise ShapeMismatchError(
                        f"Block at ({row_idx}, {col_idx}) has width {block.width}, "
                        f"but expected {width}"
                    )
            self.block_col_widths.append(width)
        
        # Вычисляем общие размеры матрицы
        self._height = sum(self.block_row_heights)
        self._width = sum(self.block_col_widths)
    
    @property
    def shape(self):
        """Возвращает размеры матрицы (height, width)."""
        return (self._height, self._width)
    
    def to_dense(self) -> np.ndarray:
        """Преобразует блочную матрицу в плотную матрицу."""
        rows = []
        for row_idx, row in enumerate(self.blocks):
            row_blocks = []
            for col_idx, block in enumerate(row):
                row_blocks.append(block.to_dense())
            rows.append(np.hstack(row_blocks))
        return np.vstack(rows)
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Умножает блочную матрицу на вектор."""
        if len(x) != self._width:
            raise ShapeMismatchError(
                f"Vector length {len(x)} does not match matrix width {self._width}"
            )
        
        result_parts = []
        x_offset = 0
        
        for row_idx, row in enumerate(self.blocks):
            row_result = None
            x_offset = 0
            
            for col_idx, block in enumerate(row):
                block_width = self.block_col_widths[col_idx]
                x_block = x[x_offset:x_offset + block_width]
                x_offset += block_width
                
                block_result = block.matvec(x_block)
                
                if row_result is None:
                    row_result = block_result
                else:
                    row_result = row_result + block_result
            
            result_parts.append(row_result)
        
        return np.concatenate(result_parts)
    
    def empty_like(self, width=None, height=None):
        """Создает пустую блочную матрицу той же структуры."""
        if width is not None or height is not None:
            raise NotImplementedError("empty_like with custom dimensions not implemented for BlockMatrix")
        
        new_blocks = []
        for row in self.blocks:
            new_row = [block.empty_like() for block in row]
            new_blocks.append(new_row)
        
        return BlockMatrix(new_blocks)
    
    @property
    def dtype(self):
        """Возвращает тип данных (наиболее общий среди всех блоков)."""
        dtypes = [block.dtype for row in self.blocks for block in row]
        return np.result_type(*dtypes)
    
    def __getitem__(self, key):
        """Получает элемент по индексу."""
        row, col = key
        
        # Находим, в каком блоке находится этот элемент
        row_offset = 0
        for row_idx, block_row in enumerate(self.blocks):
            block_height = self.block_row_heights[row_idx]
            if row < row_offset + block_height:
                col_offset = 0
                for col_idx, block in enumerate(block_row):
                    block_width = self.block_col_widths[col_idx]
                    if col < col_offset + block_width:
                        return block[row - row_offset, col - col_offset]
                    col_offset += block_width
            row_offset += block_height
        
        raise IndexError(f"Index ({row}, {col}) out of bounds")
    
    def __setitem__(self, key, value):
        """Устанавливает элемент по индексу."""
        row, col = key
        
        # Находим, в каком блоке находится этот элемент
        row_offset = 0
        for row_idx, block_row in enumerate(self.blocks):
            block_height = self.block_row_heights[row_idx]
            if row < row_offset + block_height:
                col_offset = 0
                for col_idx, block in enumerate(block_row):
                    block_width = self.block_col_widths[col_idx]
                    if col < col_offset + block_width:
                        block[row - row_offset, col - col_offset] = value
                        return
                    col_offset += block_width
            row_offset += block_height
        
        raise IndexError(f"Index ({row}, {col}) out of bounds")
    
    def __repr__(self):
        """Возвращает строковое представление блочной матрицы."""
        return f"BlockMatrix(shape={self.shape}, blocks={len(self.blocks)}x{len(self.blocks[0])})"