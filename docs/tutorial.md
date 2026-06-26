# Руководство пользователя

В этом руководстве вы научитесь создавать матрицы, выполнять с ними операции и решать линейные системы с помощью библиотеки `matlib`.

## Установка

```bash
pip install matlib-z3342-tem1-msgvk
```
## Импорт

Рекомендуется импортировать нужные классы и функции напрямую:

```python
import numpy as np
from matlib_z3342_tem1_msgvk import FullMatrix, SymmetricMatrix, BlockMatrix, solve
from matlib_z3342_tem1_msgvk import ShapeMismatchError, SingularMatrixError
```

## Создание плотной матрицы

```python
# Из numpy-массива
A = FullMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
print(A)
```

Вывод будет отформатирован в виде таблицы с границами.

## Умножение матрицы на вектор

```python
x = np.array([5.0, 6.0])
y = A.matvec(x)
print(y)  # [17. 39.]
```

## Решение системы линейных уравнений

Решим систему A x = b:

```python
A = FullMatrix(np.array([[3.0, 1.0], [1.0, 2.0]]))
b = np.array([9.0, 8.0])

x = solve(A, b)
print(x)  # [2. 3.]
```

## Работа с симметричной матрицей

```python
# Создаём симметричную матрицу (задаём только верхний треугольник)
sym = SymmetricMatrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
print(sym.to_dense())

# Умножение на вектор
v = np.array([1.0, 2.0])
res = sym.matvec(v)  # [6. 7.]
```

## Блочная матрица

Соберём блочную матрицу из нескольких блоков:

```python
A11 = FullMatrix(np.array([[1, 2], [3, 4]]))
A12 = FullMatrix(np.array([[5, 6], [7, 8]]))
A21 = FullMatrix(np.array([[9, 10], [11, 12]]))
A22 = FullMatrix(np.array([[13, 14], [15, 16]]))

B = BlockMatrix([
    [A11, A12],
    [A21, A22]
])

print(B.shape)          # (4, 4)
print(B.to_dense())     # полная матрица
```

Блочная матрица поддерживает те же операции, что и обычная: matvec, to_dense, __repr__.

## Обработка ошибок

При несовпадении размерностей или сингулярности матрицы выбрасываются соответствующие исключения:

```python
try:
    x = solve(A, b)
except SingularMatrixError:
    print("Матрица вырождена, решение невозможно.")
except ShapeMismatchError:
    print("Неверные размеры матрицы или вектора.")
```