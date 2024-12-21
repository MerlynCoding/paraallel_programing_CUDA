# Module2 Image Processing Assignment Overview Videos

### **Steps to Implement the Assignment**

#### 1. **Understand Input and Output**
   - **Input:** RGB image represented as three separate arrays of unsigned char values for red, green, and blue channels (0-255).
   - **Output:** Grayscale image represented as a single array of unsigned char values (0-255).

---

#### 2. **CUDA Kernel Implementation**
   - Each thread processes one pixel. 
   - Calculate the grayscale value for each pixel using the formula:
     \[
     \text{Gray} = \frac{\text{R} + \text{G} + \text{B}}{3}
     \]
   - Store the result in the output grayscale array.

   **Pseudocode for Kernel:**
   ```cpp
   __global__ void convertToGrayKernel(unsigned char *d_R, unsigned char *d_G, unsigned char *d_B, unsigned char *d_gray, int numPixels) {
       int idx = blockDim.x * blockIdx.x + threadIdx.x;

       if (idx < numPixels) {
           d_gray[idx] = (d_R[idx] + d_G[idx] + d_B[idx]) / 3;
       }
   }
   ```

---

#### 3. **Host Code**
   - Load the image data into RGB arrays.
   - Allocate device memory and copy the input arrays (RGB) from the host to the device.
   - Launch the CUDA kernel with appropriate grid and block configurations.
   - Copy the grayscale output from the device back to the host.
   - Save the resulting grayscale image and calculate the mean difference percentage.

---

#### 4. **Image Comparison**
   - Use the CPU-based implementation for comparison.
   - Calculate the mean difference percentage between the CPU and GPU results for all pixels:
     \[
     \text{Mean Difference Percentage} = \frac{\sum_{i=0}^{n-1} |\text{Gray}_{\text{CPU}, i} - \text{Gray}_{\text{GPU}, i}|}{n}
     \]

---

#### 5. **Evaluation Criteria**
   - The closer the GPU results are to the CPU results, the higher the score.
   - You can get:
     - **80% score** by implementing the basic averaging algorithm.
     - **95-100% score** by addressing issues like row/column boundary mismatches or memory alignment discrepancies.

---

### **Code Snippets**

#### Kernel Launch in Host Code:
```cpp
dim3 block(256);
dim3 grid((numPixels + block.x - 1) / block.x);

convertToGrayKernel<<<grid, block>>>(d_R, d_G, d_B, d_gray, numPixels);
cudaDeviceSynchronize();
```

#### Copy Results Back to Host:
```cpp
cudaMemcpy(h_gray, d_gray, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
```

#### Save Output Image:
Use OpenCV or another library to write the resulting grayscale image to a file.

---

### **Debugging Tips**
1. Ensure that the image dimensions align with thread and block layout. Handle edge cases where the number of threads may exceed the number of pixels.
2. Use **cudaMemcpy** and **cudaMemcpyToSymbol** correctly for data transfer.
3. Check for CUDA errors with `cudaGetLastError()` and `cudaDeviceSynchronize()` after the kernel execution.

---

### **Extra Challenges**
- Handle images with different memory layouts efficiently.
- Optimize for memory coalescing to improve GPU performance.
- Experiment with using shared memory for larger image processing tasks.

---

By implementing the kernel and following the instructions, you should achieve the expected output and score well on the assignment. If you have specific questions about implementation or debugging, let me know!
