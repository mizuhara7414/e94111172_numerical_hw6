import numpy as np


A = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],  
    [15.3, 0.110, -13.1, -1]
])

b = np.array([1.12, 3.44, 2.15, 4.16])

# 增廣矩陣
augmented = np.column_stack((A, b))
print("增廣矩陣:")
print(augmented)


n = len(b)
for i in range(n):
    
    max_row = i + np.argmax(abs(augmented[i:, i]))
    if max_row != i:
        augmented[[i, max_row]] = augmented[[max_row, i]]
    
    # eliminate
    for j in range(i+1, n):
        factor = augmented[j, i] / augmented[i, i]
        augmented[j, i:] -= factor * augmented[i, i:]

print("\n上三角:")
print(augmented)

# find x
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = (augmented[i, -1] - np.sum(augmented[i, i+1:n] * x[i+1:])) / augmented[i, i]

print("\n x:")
print(x)

# 
result = A @ x
print("\n A*x:")
print(result)
print(" b:")
print(b)