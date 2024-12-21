# Threads, Blocks, and Grids

### **Key Points:**

1. **Dimensionality:**
   - **1D Threads and Blocks:** Simplest case, suitable for linear arrays. Blocks contain threads indexed by `threadIdx.x`, and the grid index is accessed via `blockIdx.x`.
   - **2D Threads and Blocks:** Adds a second dimension, ideal for matrices. Threads and blocks are accessed via `threadIdx.x`, `threadIdx.y`, and `blockIdx.x`, `blockIdx.y`.
   - **3D Threads and Blocks:** Adds a third dimension for volumetric data like 3D arrays. Uses `threadIdx.z`, `blockIdx.z`, etc.

2. **Dim3 Types:**
   - **`dim3` Variables:** Used to define grid and block layouts in three dimensions:
     ```cpp
     dim3 threadsPerBlock(16, 16, 16); // Threads in X, Y, Z
     dim3 blocksPerGrid(2, 2, 2); // Blocks in X, Y, Z
     ```

3. **Calculating Thread Index:**
   - The global thread index is calculated using:
     ```cpp
     int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
     ```
   - For 2D layouts:
     ```cpp
     int x = threadIdx.x + blockIdx.x * blockDim.x;
     int y = threadIdx.y + blockIdx.y * blockDim.y;
     int globalIdx = y * gridDim.x * blockDim.x + x;
     ```
   - For 3D layouts:
     ```cpp
     int x = threadIdx.x + blockIdx.x * blockDim.x;
     int y = threadIdx.y + blockIdx.y * blockDim.y;
     int z = threadIdx.z + blockIdx.z * blockDim.z;
     int globalIdx = z * gridDim.y * gridDim.x * blockDim.y * blockDim.x +
                     y * gridDim.x * blockDim.x +
                     x;
     ```

4. **Resiliency:**
   - This indexing strategy works regardless of grid and block dimensions, making the kernel flexible for various configurations.

---

### **Sample Code for Multi-Dimensional Thread and Block Layouts**

#### **Kernel Example for 1D Layout**
```cpp
__global__ void process1D(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = idx * idx; // Example operation
    }
}
```

#### **Kernel Example for 2D Layout**
```cpp
__global__ void process2D(int *data, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * width + x; // Row-major index
    if (x < width && y < height) {
        data[idx] = x + y; // Example operation
    }
}
```

#### **Kernel Example for 3D Layout**
```cpp
__global__ void process3D(int *data, int width, int height, int depth) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = z * width * height + y * width + x; // 3D index
    if (x < width && y < height && z < depth) {
        data[idx] = x * y * z; // Example operation
    }
}
```

---

### **Host Code to Launch Kernels**

#### **1D Launch:**
```cpp
int n = 1024;
dim3 threadsPerBlock(256);
dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
process1D<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
```

#### **2D Launch:**
```cpp
int width = 32, height = 32;
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
process2D<<<blocksPerGrid, threadsPerBlock>>>(d_data, width, height);
```

#### **3D Launch:**
```cpp
int width = 16, height = 16, depth = 16;
dim3 threadsPerBlock(8, 8, 8);
dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);
process3D<<<blocksPerGrid, threadsPerBlock>>>(d_data, width, height, depth);
```

---

### **Key Takeaways**
- **Generalize Your Kernel:** By calculating global thread indices using `threadIdx` and `blockIdx` across all dimensions, your code becomes adaptable to different layouts.
- **Boundary Checks:** Always ensure your thread indices are within bounds to avoid memory access errors.
- **Use `dim3` Wisely:** Define threads and blocks in `dim3` types for clarity and scalability.

This approach ensures flexibility and compatibility with varying dimensional data layouts.
