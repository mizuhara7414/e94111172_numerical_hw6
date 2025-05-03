import numpy as np

# 定義矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

print(" A:")
print(A)



n = A.shape[0]
I = np.array([[1, 0], [0, 1]], dtype=np.float64)
ide = np.kron(I,I)
Aug = np.column_stack((A,ide ))

print("\n增廣矩陣 [A|I]:")
print(Aug)


for i in range(n):
   
    max_row = i + np.argmax(abs(Aug[i:, i]))
    if max_row != i:
        Aug[[i, max_row]] = Aug[[max_row, i]]
    
    
    Aug[i] = Aug[i] / Aug[i, i]
    
    
    for j in range(n):
        if j != i:
            Aug[j] = Aug[j] - Aug[j, i] * Aug[i]

print("\ngaussian - jordan後的矩陣:")
print(Aug)

# 逆矩陣部分
A_inv = Aug[:, n:2*n]

print("\ninverse matrix A^(-1):")
print(A_inv)

# 驗證 A * A^(-1) = I
print("\n A * A^(-1):")
print(np.dot(A, A_inv))

