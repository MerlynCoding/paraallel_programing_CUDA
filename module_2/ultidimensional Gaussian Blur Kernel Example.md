# ultidimensional Gaussian Blur Kernel Example

### **Key Concepts**

#### 1. **3D Gaussian Blur:**
   - **Purpose:** Smooths variations in data, reducing noise while retaining edges.
   - **Operation:**
     - Takes a pixel (x, y) in an image (grayscale value: 0-100).
     - Blurs the pixel based on a 3x3x3 kernel that spans:
       - Neighboring pixels in the same frame.
       - Corresponding pixels in the previous and next frames (temporal blur).
   - **Padding:** Adds a 1-pixel border and 1 frame before and after for boundary handling.

#### 2. **Thread and Block Layout:**
   - **3D Threads:** Operate on x, y, z dimensions.
   - **3D Blocks:** Process sub-volumes (blocks of pixels).
   - **Grid Layout:** Maps to frames, ensuring spatial and temporal processing.

#### 3. **Kernel Design:**
   - **Input:** 3D array of pixel values representing video.
   - **Output:** Blurred video.
   - **Mask:** Gaussian weights, applied based on distance from the center of the kernel.

---

### **Steps in Implementation**

#### **Host Code:**
1. **Define Input/Output Memory:**
   - Create 3D arrays for video data and the blurred output.
   - Allocate GPU memory and copy input data.

2. **Define Kernel Layout:**
   - Use `dim3` to define threads per block and blocks per grid.
     ```cpp
     dim3 threadsPerBlock(16, 16, 1); // Threads for x, y, z
     dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, numFrames);
     ```

3. **Launch Kernel:**
   - Execute the kernel with specified layouts.
     ```cpp
     gaussianBlur3D<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height, numFrames);
     ```

4. **Copy Results:**
   - Transfer blurred video back to host memory.

---

#### **Device Code:**

##### **3D Slice Function:**
- Extracts a 3x3x3 cube of data centered around a pixel.
- Handles edge cases using padding.

**Pseudocode:**
```cpp
__device__ float get3DSlice(const float *input, int x, int y, int z, int width, int height, int depth) {
    float slice[3][3][3];
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                slice[dz + 1][dy + 1][dx + 1] = input[clamp(z + dz, 0, depth - 1) * width * height +
                                                     clamp(y + dy, 0, height - 1) * width +
                                                     clamp(x + dx, 0, width - 1)];
            }
        }
    }
    return slice;
}
```

##### **3D Gaussian Blur Kernel:**
- Applies a Gaussian mask to the 3D slice to compute a weighted mean.
- Stores the result in the output array.

**Pseudocode:**
```cpp
__global__ void gaussianBlur3D(const float *input, float *output, int width, int height, int depth) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < width && y < height && z < depth) {
        float sum = 0;
        float weightSum = 0;

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = clamp(x + dx, 0, width - 1);
                    int ny = clamp(y + dy, 0, height - 1);
                    int nz = clamp(z + dz, 0, depth - 1);

                    float weight = gaussianWeight(dx, dy, dz);
                    sum += input[nz * width * height + ny * width + nx] * weight;
                    weightSum += weight;
                }
            }
        }

        output[z * width * height + y * width + x] = sum / weightSum;
    }
}
```

##### **Gaussian Weight Function:**
Calculates weights based on distance from the center pixel.

**Pseudocode:**
```cpp
__device__ float gaussianWeight(int dx, int dy, int dz) {
    float sigma = 1.0;
    return expf(-(dx * dx + dy * dy + dz * dz) / (2 * sigma * sigma));
}
```

---

### **Key CUDA Concepts Used**
- **Memory Allocation:** Use `cudaMalloc` and `cudaMemcpy` to manage GPU memory.
- **Thread Indexing:** Calculate indices in 3D space using `threadIdx` and `blockIdx`.
- **Boundary Checks:** Use `clamp` to handle edge cases when accessing neighbors.
- **Dim3 Layout:** Define threads and blocks in three dimensions for spatial and temporal processing.

---

### **Performance Considerations**
- **Thread Occupancy:** Optimize `threadsPerBlock` and `blocksPerGrid` for GPU hardware.
- **Shared Memory:** Use shared memory to reduce global memory access latency.
- **Memory Coalescing:** Ensure threads access contiguous memory to maximize throughput.

---

This approach effectively applies a 3D Gaussian blur to video data, leveraging CUDA's parallel processing capabilities for efficient computation.
