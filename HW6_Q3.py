import numpy as np

# 定義矩陣 A 和向量 b
A = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
], dtype=float)

b = np.array([2, 3, 4, 1], dtype=float)

print(" A:")
print(A)
print("\n b:")
print(b)

#  Crout 
n = A.shape[0]

#  L 和 U 
L = np.zeros((n, n))
U = np.zeros((n, n))


for i in range(n):
    U[i, i] = 1

# Crout 分解計算 L 和 U
for i in range(n):
    # 計算 L 的第 i 列
    for j in range(i, n):
        sum_lu = sum(L[j, k] * U[k, i] for k in range(i))
        L[j, i] = A[j, i] - sum_lu
    
    # 計算 U 的第 i 行（對角線以上）
    for j in range(i+1, n):
        if L[i, i] == 0:
            raise ValueError("error")
        sum_lu = sum(L[i, k] * U[k, j] for k in range(i))
        U[i, j] = (A[i, j] - sum_lu) / L[i, i]

print("\nL matrix:")
print(L)
print("\nU matrix:")
print(U)

# 前向替換求解 Ly = b
y = np.zeros(n)
for i in range(n):
    y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]


# 後向替換求解 Ux = y
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))

print("\n x:")
print(x)

# 驗證結果
print("\n A*x:")
print(np.dot(A, x))
print(" b:")
print(b)

