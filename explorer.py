import numpy as np
import time
import sys

print("=== NumPy Data Explorer Project ===\n")

# 1. NumPy Fundamentals: Array Creation, Indexing, Slicing
print("1. Array Creation, Indexing, Slicing")
arr1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Original Array:\n", arr1)

# Indexing
print("\nElement at row 1, col 2:", arr1[1, 2])  # Output: 7

# Slicing
print("First 2 rows, last 2 columns:\n", arr1[0:2, 2:4])
print("Every 2nd row:\n", arr1[::2, :])

# 2. Mathematical, Axis-wise, and Statistical Operations
print("\n2. Mathematical & Statistical Operations")
arr2 = np.random.randint(1, 20, size=(4, 5))
print("Random Array:\n", arr2)

print("\nSum of all elements:", np.sum(arr2))
print("Mean:", np.mean(arr2))
print("Std Deviation:", np.std(arr2))
print("Sum per column (axis=0):", np.sum(arr2, axis=0))
print("Mean per row (axis=1):", np.mean(arr2, axis=1))
print("Max value:", np.max(arr2), "at position:", np.unravel_index(np.argmax(arr2), arr2.shape))

# 3. Reshaping and Broadcasting
print("\n3. Reshaping & Broadcasting")
arr3 = np.arange(12)
print("Original 1D array:", arr3)

reshaped = arr3.reshape(3, 4)
print("Reshaped to 3x4:\n", reshaped)

# Broadcasting example: add a 1D array to each row
row_to_add = np.array([10, 20, 30, 40])
broadcasted = reshaped + row_to_add
print("After broadcasting + row_to_add:\n", broadcasted)

# 4. Save and Load Operations
print("\n4. Save/Load Operations")
np.save("my_array.npy", arr2)
loaded_arr = np.load("my_array.npy")
print("Array saved and loaded successfully. Loaded array:\n", loaded_arr)

# 5. Performance Comparison: NumPy vs Python List
print("\n5. Performance Comparison: NumPy vs Python List")
size = 1_000_000

# Python list
py_list = list(range(size))
start = time.time()
py_result = [x * 2 for x in py_list]
py_time = time.time() - start

# NumPy array
np_arr = np.arange(size)
start = time.time()
np_result = np_arr * 2
np_time = time.time() - start

print(f"Python list time: {py_time:.5f} seconds")
print(f"NumPy array time: {np_time:.5f} seconds")
print(f"NumPy is {py_time/np_time:.2f}x faster!")

# Memory usage
print(f"\nMemory usage - Python list: {sys.getsizeof(py_list) / 1024:.2f} KB")
print(f"Memory usage - NumPy array: {np_arr.nbytes / 1024:.2f} KB")

print("\n=== Project Completed ===")