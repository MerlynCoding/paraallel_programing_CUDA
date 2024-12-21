# Week 2 Programming Lab Guide: Search Kernel Implementation

#### Overview
The goal of this lab is to implement and execute a CUDA kernel that searches for a specified value within a dataset. Below is a detailed breakdown of the Makefile, the main program structure, the search kernel, and the key functions involved.

---

### **1. The Makefile**
The `Makefile` supports three commands:
- **`build`**: Compiles the CUDA program into an executable (`search.exe`).
- **`clean`**: Removes any compiled binaries (`search.exe`).
- **`run`**: Executes the `search.exe` program.

To use the `Makefile`:
```bash
make build    # Builds the program
make clean    # Cleans up the binaries
make run      # Runs the program
```

---

### **2. Program Execution**
When you run the program, it performs the following:
1. Generates or reads input data (random numbers or from a CSV).
2. Allocates memory on the host (CPU) and the device (GPU).
3. Copies input data from the host to the GPU device.
4. Executes the **search kernel** on the GPU.
5. Copies the results from the GPU back to the host.
6. Outputs the search results, including the input data, search value, and the index of the found value.

---

### **3. Key Functions and Their Roles**
#### **Main Function**
- **Command-line arguments**:
  - Input data (CSV file or random).
  - Sorting (optional).
  - Number of threads per block.
  - The search value.
- **Main responsibilities**:
  1. Parse input arguments.
  2. Initialize and allocate memory.
  3. Copy data from host to device.
  4. Call the kernel.
  5. Copy results from device to host.
  6. Output results.

#### **Kernel (`search`)**
- **Purpose**: Searches for a specific value in the dataset.
- **Inputs**:
  - `d_d`: Device pointer to the data array.
  - `d_i`: Device pointer to the output index.
  - `numElements`: Total number of elements in the data array.
- **Logic**:
  1. Calculate the thread's index.
  2. Verify the index is within bounds.
  3. Compare the thread's assigned value with the search value.
  4. If found, write the index to `d_i`.

#### **`allocateRandomMemory`**
- **Purpose**: Generates random input data.
- **Inputs**: None.
- **Outputs**: Allocates and initializes random numbers in host memory.

#### **`copyFromHostToDevice`**
- **Purpose**: Copies host data arrays (`h_d`, `h_i`) to device arrays (`d_d`, `d_i`).
- **Logic**: Uses `cudaMemcpy`.

#### **`executeKernel`**
- **Purpose**: Manages the execution of the CUDA kernel.
- **Logic**:
  1. Determines grid/block size.
  2. Launches the kernel.
  3. Handles kernel errors.

#### **`copyFromDeviceToHost`**
- **Purpose**: Copies results from device memory back to host memory.
- **Logic**: Uses `cudaMemcpy`.

#### **`deallocateMemory`**
- **Purpose**: Frees memory allocated on the GPU and CPU.

#### **`cleanupDevice`**
- **Purpose**: Resets the GPU state.

---

### **4. Key Kernel Code**
```cpp
__global__ void search(const int *d_d, int *d_i, int numElements, int searchValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements && d_d[idx] == searchValue) {
        d_i[0] = idx;  // Store the index of the found value
    }
}
```

---

### **5. Expected Output**
After running `search.exe`, you should see:
- The input data (random or CSV).
- The search value.
- The index where the search value is found (e.g., `9`).

**Example Output:**
```plaintext
Input Data: [12, 7, 22, 9, 5]
Search Value: 9
Found at Index: 3
```

---

### **6. Tips for Success**
1. **Understand the Kernel Execution**:
   - Blocks and threads define the granularity of search.
   - Ensure that the grid/block configuration covers all data points.

2. **Error Handling**:
   - Use `cudaGetLastError()` after kernel calls to catch issues.
   - Check memory allocation results on the GPU.

3. **Optimize Threads/Blocks**:
   - Choose thread counts that balance workload across blocks.

4. **Use Debugging Tools**:
   - `cuda-gdb` for kernel debugging.
   - `cuda-memcheck` to ensure proper memory usage.

---

### **7. Further Exploration**
- Experiment with large datasets to understand GPU scalability.
- Modify the kernel to store all found indices instead of overwriting.
- Test edge cases where the search value does not exist.

This lab introduces you to foundational CUDA programming concepts and prepares you for more complex GPU computing tasks.
