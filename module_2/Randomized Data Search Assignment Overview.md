# Randomized Data Search Assignment Overview

This assignment focuses on searching through randomized data using a CUDA kernel. Below is a step-by-step breakdown of the task and what is required to successfully complete and submit the assignment.

---

### **1. Overview**
The assignment uses randomized or pre-defined data to search for a specific value. You will work with a CUDA kernel (`search kernel`) and modify it to achieve the desired functionality.

---

### **2. Command-Line Arguments**
The program takes several command-line arguments:
- **Data options**:
  - `--sorted`: Whether the data should be sorted.
  - `--num-elements`: The number of elements to generate.
  - `--search-value`: The value to search for.
  - `--input-file`: A file containing predefined input data.
- **Execution parameters**:
  - `--threads-per-block`: The number of threads per block.
  - **Note**: `num-elements` should always be greater than `threads-per-block`.

---

### **3. Workflow**
1. **Prepare the Project**:
   - Navigate to the project folder.
   - Modify the search kernel as required.

2. **Build the Project**:
   - Use the command:
     ```bash
     make clean build
     ```
   - This cleans previous builds and compiles the project.

3. **Run the Program**:
   - Execute the program using:
     ```bash
     ./search.exe
     ```
   - Depending on the input, it will:
     - Generate randomized data or read input from a file.
     - Search for the specified value.

4. **Observe the Output**:
   - The output will include:
     - **Input Data**: The dataset being searched.
     - **Search Value**: The value being searched for.
     - **Index Found**: The index of the search value in the dataset (if found).

5. **Debugging**:
   - Add additional print statements during debugging, but **remove them before submission**.

---

### **4. Kernel Explanation**
The search kernel is the core part of this assignment. It handles:
1. **Index Calculation**:
   - Computes the thread's index in the dataset.
2. **Boundary Check**:
   - Ensures the index is within the dataset size.
3. **Value Comparison**:
   - Compares the thread's assigned value to the search value.
   - Writes the thread index to the result if the value matches.

---

### **5. Submission**
1. **Generate Output Files**:
   - After running, several output files (`output-partID.txt`) will be created.
   - These files contain:
     - Input Data.
     - Search Value.
     - Search Result.

2. **Submit the Assignment**:
   - Use the **Submit** button in the interface.
   - Submissions are graded on:
     - Correctness of the output.
     - Completion of all parts.

---

### **6. Example Kernel**
Below is a simple example of a CUDA search kernel:
```cpp
__global__ void searchKernel(const int *data, int *result, int numElements, int searchValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements && data[idx] == searchValue) {
        result[0] = idx;  // Write the index to result
    }
}
```

---

### **7. Key Notes**
- Ensure the output format matches the required structure.
- Do not modify predefined print statements or provided data generation methods.
- Check for memory allocation and kernel execution errors.

---

### **8. Common Issues**
- **Kernel Errors**:
  - Use `cudaGetLastError()` to debug kernel issues.
- **Memory Issues**:
  - Ensure device memory is correctly allocated and freed.
- **Output Format**:
  - Ensure the output matches the required format for grading.

This assignment helps reinforce concepts of CUDA memory management, kernel execution, and parallel search algorithms.
