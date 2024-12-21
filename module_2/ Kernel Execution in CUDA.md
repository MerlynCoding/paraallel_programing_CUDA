# Kernel Execution in CUDA - Lecture Summary

Welcome to the **Kernel Execution** lecture as part of the *Introduction to Parallel Programming with CUDA*. In this lecture, we explore the high-level GPU architecture, the mapping between hardware and software, and the core concepts of kernel execution in CUDA programming.

---

### **1. High-Level GPU Architecture**

The **GPU architecture** we’ll discuss is based on NVIDIA's *Ampere* architecture:

- **Core Components**:
  - **Streaming Multiprocessors (SMs)**: The heart of GPU computation, responsible for executing CUDA threads.
  - **Memory Controllers**: Manage memory transfers between different memory types on the GPU.
  - **GigaThread Engine**: The main scheduler that dispatches threads for execution.
  - **Interfaces**:
    - **General IO Bus** (top): Shared between CPU and GPU, handles data transfer.
    - **NVLink** (bottom): High-speed link for faster data transmission between GPUs.

---

### **2. Streaming Multiprocessor (SM) Overview**

An SM is a unit within the GPU containing:

- **Warps**: A collection of 32 threads.
  - Each warp includes:
    - **Scheduler**: Manages the execution of instructions.
    - **Register File**: Stores thread-specific data.
    - Two **half-warps**:
      - One for **floating-point** or **integer** operations.
      - One dedicated to **32-bit floating-point** operations.
    - **Tensor Core**: Optimized for deep learning and matrix operations.
    - **L1 Data Store**: Acts as shared memory between threads.
    - **Texture Cores**: Handle graphics data.
    - **Ray Tracing Core**: Specialized for rendering graphics.

---

### **3. Mapping Software to Hardware in CUDA**

In CUDA, the computational hierarchy is as follows:

1. **Thread**:
   - Smallest computational unit.
   - Executes a single kernel instruction on a core.

2. **Block**:
   - A collection of threads (commonly 32 threads per warp).
   - Maps to partitions within an SM.

3. **Grid**:
   - Holds multiple thread blocks.
   - Can span across multiple GPUs for large-scale computations.

---

### **4. Kernel Execution in CUDA**

A CUDA **kernel** is a function executed on the GPU. Its configuration determines how the workload is divided across the GPU.

#### **Kernel Launch Syntax**
```cpp
kernelName<<<Dg, Db, sharedMem, stream>>>(kernelArgs);
```
- `Dg` (Grid Dimensions): Specifies the number of blocks in the grid (can be 1D, 2D, or 3D).
- `Db` (Block Dimensions): Specifies the number of threads per block (can also be 1D, 2D, or 3D).
- `sharedMem` (optional): Amount of shared memory per block.
- `stream` (optional): CUDA stream for concurrent kernel execution.

#### **Example:**
```cpp
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
```

---

### **5. Kernel Definition**

A CUDA kernel function is defined using the `__global__` qualifier and must return `void`. 

#### **Example Kernel:**
```cpp
__global__ void addMatrices(float *A, float *B, float *C, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = j * width + i;
    if (i < width && j < height) {
        C[index] = A[index] + B[index];
    }
}
```

#### Key Components:
- **Thread and Block IDs**:
  - `blockIdx`: Block ID within the grid.
  - `threadIdx`: Thread ID within a block.
  - `blockDim`: Dimensions of a block (threads per block).
  - `gridDim`: Dimensions of the grid (blocks per grid).

- **Index Calculation**:
  - Convert 2D block and thread indices into a single linear index to map to arrays.

---

### **6. 2D and 3D Thread and Block Layout**

CUDA supports **multi-dimensional thread and block configurations**, enabling efficient mapping to problems involving matrices or 3D data (e.g., video processing).

- **1D Example**:
  ```cpp
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  ```

- **2D Example**:
  ```cpp
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  ```

- **3D Example**:
  ```cpp
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  ```

---

### **7. Shared and Device Memory in Kernel Execution**

- **Shared Memory**:
  - Memory shared by all threads within a block.
  - Faster than global memory.

- **Global Memory**:
  - Accessible by all threads but slower compared to shared memory.

- **Registers**:
  - Store private data for individual threads.

---

### **Key Takeaways**

1. CUDA threads, blocks, and grids provide a structured way to distribute computational workloads on NVIDIA GPUs.
2. Kernels define the computation logic and are executed by threads on GPU cores.
3. Proper configuration of grid and block dimensions is crucial for optimal performance.
4. Shared memory and device memory are essential for efficient data handling in CUDA programs.

This concludes the **Kernel Execution** lecture. You're now equipped with foundational knowledge to start writing and optimizing CUDA kernels!
