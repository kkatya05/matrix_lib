# API Reference

## Классы матриц

### 'FullMatrix'

Плотная (заполненная) матрица, хранящая данные в виде `numpy.ndarray`.

**Конструктор**  
`FullMatrix(data: np.ndarray)`

**Атрибуты**  
- `shape` – кортеж `(height, width)`  
- `dtype` – тип данных элементов

**Основные методы**  
- `to_dense() -> np.ndarray` – возвращает копию внутреннего массива.  
- `matvec(x: np.ndarray) -> np.ndarray` – умножение на вектор `x`.  
- `empty_like(width=None, height=None) -> FullMatrix` – создаёт пустую матрицу с такими же параметрами.  
- `__getitem__`, `__setitem__` – доступ по индексам `[row, col]`.

**Исключения**  
- `ShapeMismatchError` – если размеры не совпадают при умножении.

**Пример**  
```python
A = FullMatrix(np.array([[1, 2], [3, 4]]))
x = np.array([1, 1])
print(A.matvec(x))  
```
Выводит: [3, 7]

### 'SymmetricMatrix

Симметричная матрица; хранится только верхний треугольник.

**Конструктор**  

`SymmetricMatrix(data: np.ndarray)` 
`data` должна быть квадратной; сохраняется как есть, но доступ к нижнему треугольнику перенаправляется на верхний.

**Атрибуты** 

- `shape` – кортеж `(height, width)`  
- `dtype` – тип данных элементов

**Основные методы**  

- `to_dense() -> np.ndarray` – восстанавливает полную матрицу..  
- `matvec(x: np.ndarray) -> np.ndarray` – умножение на вектор `x`.  
- `empty_like(width=None, height=None) -> FullMatrix` – создаёт пустую матрицу с такими же параметрами.  

**Исключения**  

- `MatrixNotSquareError` – если входной массив не квадратный.

**Пример** 

```python
S = SymmetricMatrix(np.array([[2, 1], [1, 2]]))
print(S[1, 0])  # 1 (читается из верхнего треугольника)
```

### 'BlockMatrix'

Блочная матрица, составленная из прямоугольной таблицы блоков (любых объектов, наследующих Matrix).

**Конструктор** 

`BlockMatrix(blocks: List[List[Matrix]])`

**Атрибуты** 

- `shape` – кортеж `(height, width)`  
- `dtype` – тип данных элементов

**Основные методы**  

- `to_dense() -> np.ndarray` – объединяет блоки в один массив..  
- `matvec(x: np.ndarray) -> np.ndarray` – умножение на вектор `x`, разбивая его по блокам..  
- `empty_like()` – создаёт блочную матрицу той же структуры с пустыми блоками.  

**Исключения**  

- `ShapeMismatchError` – если размеры блоков не согласованы или вектор не подходит.

**Пример**  

```python
B = BlockMatrix([[A11, A12], [A21, A22]])
print(B.shape)
```

### 'Решение СЛАУ'

`solve(A: Matrix, b: np.ndarray) -> np.ndarray` – Решает систему `Ax = b` методом PLU-раздожения.

**Параметры**

- `А` – объект матрицы (любого типа, поддерживающий to_dense()).  
- `b` – вектор (1D) или матрица (2D) правой части. Число строк b должно совпадать с числом строк А.

**Возвращает**

- `np.ndarray` - решение `x`.

**Исключения**

- `MatrixNotSquareError` – если матрица не квадратная.
- `ShapeMismatchError` – если размерности не согласованы.
- `SingularMatrixError` – если матрица вырождена.

**Пример**
```python
A = FullMatrix(np.array([[2, 1], [1, 3]]))
b = np.array([5, 6])
x = solve(A, b)  # [1.8, 1.4]
```

