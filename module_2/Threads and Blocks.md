# Threads and Blocks

---

### **1. One-Dimensional Thread Layout**
- **Kernel Launch:**
  ```cpp
  kernel_name<<<1, threadsPerBlock>>>(args);
  ```
  - `<<<1, threadsPerBlock>>>`: One block with `threadsPerBlock` threads.
- **Thread Index Calculation:**
  ```cpp
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ```
  - `blockIdx.x`: Block index (always 0 here, as only 1 block is used).
  - `blockDim.x`: Total threads per block.
  - `threadIdx.x`: Current thread index within the block.

- **Boundary Check:**
  Always ensure:
  ```cpp
  if (idx < numElements) {
      // Safe to access data
  }
  ```
---

### **2. Two-Dimensional Thread Layout**
- **Kernel Launch:**
  ```cpp
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid(1, 1);
  kernel_name<<<blocksPerGrid, threadsPerBlock>>>(args);
  ```
  - `dim3 threadsPerBlock(32, 32)`: A 2D layout of 32x32 threads in each block.
  - `dim3 blocksPerGrid(1, 1)`: Single block in the grid.
  
- **Thread Index Calculation:**
  ```cpp
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  ```
  - `blockIdx.x/y`: Block indices in x and y dimensions.
  - `blockDim.x/y`: Threads per block in x and y dimensions.
  - `threadIdx.x/y`: Thread indices within the block.

- **Boundary Check:**
  Ensure:
  ```cpp
  if (x < width && y < height) {
      // Safe to access matrix
  }
  ```

---

### **3. Three-Dimensional Thread Layout**
- **Kernel Launch:**
  ```cpp
  dim3 threadsPerBlock(16, 16, 16);
  dim3 blocksPerGrid(1, 1, 1);
  kernel_name<<<blocksPerGrid, threadsPerBlock>>>(args);
  ```
  - `dim3 threadsPerBlock(16, 16, 16)`: A 3D layout of 16x16x16 threads.
  - `dim3 blocksPerGrid(1, 1, 1)`: Single block in the grid.

- **Thread Index Calculation:**
  ```cpp
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  ```
  - Add the respective offsets for `x`, `y`, and `z`.

- **Boundary Check:**
  Ensure:
  ```cpp
  if (x < width && y < height && z < depth) {
      // Safe to access 3D data
  }
  ```

---

### **4. General Guidelines**
- **Use `dim3` for Flexibility:**
  - `dim3 threadsPerBlock` and `dim3 blocksPerGrid` allow you to specify multi-dimensional thread and block layouts.
- **Match Thread Layout to Data:**
  - Choose a 1D, 2D, or 3D layout based on the problem structure (e.g., arrays, matrices, or volumes).
- **Boundary Checks:**
  - Always verify that calculated indices are within data bounds to avoid out-of-bound memory access.
- **Use Shared Memory for Optimization:**
  - If threads within a block share data, utilize shared memory to reduce global memory access.

---

### Example: Matrix Multiplication Kernel
```cpp
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        float value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[y * N + i] * B[i * N + x];
        }
        C[y * N + x] = value;
    }
}
```
- **Launch Configuration:**
  ```cpp
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
  matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  ```

This example demonstrates combining kernel thread and block definitions with efficient indexing and boundary checks.
