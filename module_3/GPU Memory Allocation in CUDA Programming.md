# GPU Memory Allocation in CUDA Programming

This README introduces the concepts and practical usage of host memory allocation in CUDA programming. It explains the types of memory, their purposes, and how to use them with simple code examples.

---

## 1. What is Host Memory Allocation?

Before data can be used in a GPU, it must be allocated in host memory (on the CPU). CUDA provides various ways to manage this memory, depending on your performance needs.

---

## 2. Types of Host Memory in CUDA

### 2.1 Pageable Memory
- **Default Memory Type**: Allocated like regular CPU memory.
- **Performance**: Slowest, as it requires an extra copy during GPU transfers.
- **Example**:
  ```cpp
  int *array = (int *)malloc(size * sizeof(int)); // Standard C allocation
  free(array);
  ```

### 2.2 Pinned Memory
- **Direct Transfer**: Saves time by avoiding extra copy steps.
- **Usage**: Use `cudaMallocHost` to allocate pinned memory.
- **Example**:
  ```cpp
  cudaMallocHost((void **)&array, size * sizeof(int));
  cudaFreeHost(array);
  ```

### 2.3 Mapped Memory
- **Shared Access**: Allows GPU to access memory directly from the CPU.
- **Usage**: Use `cudaHostMalloc` with the `cudaHostAllocMapped` flag.
- **Example**:
  ```cpp
  cudaHostMalloc((void **)&array, size * sizeof(int), cudaHostAllocMapped);
  cudaFreeHost(array);
  ```

### 2.4 Unified Memory
- **Unified View**: Treats host and GPU memory as a single memory space.
- **Usage**: Use `cudaMallocManaged` for simplified memory management.
- **Example**:
  ```cpp
  cudaMallocManaged(&array, size * sizeof(int));
  cudaDeviceSynchronize();
  cudaFree(array);
  ```

---

## 3. Practical Examples

### Example 1: Unified Memory with a Kernel
This example shows how to use unified memory with a CUDA kernel.

```cpp
#include <cuda_runtime.h>
#include <iostream>

// GPU Kernel
__global__ void kernel(int *array, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        array[idx] += 10; // Add 10 to each element
    }
}

int main() {
    int size = 10;
    int *array;

    // Allocate unified memory
    cudaMallocManaged(&array, size * sizeof(int));

    // Initialize array
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }

    // Launch kernel
    kernel<<<1, size>>>(array, size);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Print results
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }

    // Free unified memory
    cudaFree(array);

    return 0;
}
```

---

### Example 2: Pinned Memory for Faster Transfers
This example demonstrates pinned memory allocation for efficient host-to-GPU transfers.

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int size = 10;
    int *array;

    // Allocate pinned memory
    cudaMallocHost((void **)&array, size * sizeof(int));

    // Initialize array
    for (int i = 0; i < size; i++) {
        array[i] = i * 2;
    }

    // Free pinned memory
    cudaFreeHost(array);

    return 0;
}
```

---

## 4. Key Takeaways
1. **Choose the Right Memory Type**:
   - **Pageable**: Default but slow for GPU transfers.
   - **Pinned**: Faster for host-to-GPU transfers.
   - **Mapped**: Direct GPU access to host memory.
   - **Unified**: Simplifies memory management.
2. **Practice Syntax**: Use functions like `cudaMallocHost`, `cudaHostMalloc`, and `cudaMallocManaged` correctly.
3. **Synchronize with GPU**: Always use `cudaDeviceSynchronize` after kernel execution when using unified memory.

---

## 5. Additional Resources
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [C++ Memory Allocation Tutorial](https://www.learncpp.com/)
- [Beginner’s Guide to CUDA Programming](https://developer.nvidia.com/cuda-zone)
