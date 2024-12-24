# How to Identify GPU Memory and Capabilities

This README will guide you on how to find information about your NVIDIA GPU, including global memory, product details, and more. It also introduces useful resources to get more detailed specifications about your GPU.

---

## 1. Using the `nvidia-smi` Command

The `nvidia-smi` command is your primary tool for identifying the details of your GPU connected to the system.

### Key Steps:
1. Open a terminal.
2. Run the command:
   ```bash
   nvidia-smi
   ```
3. Look for the following in the output:
   - **Product Name**: The specific GPU model (e.g., Tesla C2075).
   - **Memory Information**: Available global memory and usage statistics.
   - **Other Information**: Utilization, temperature, and more.

### Example Output:
- Product Name: Tesla C2075
- Global Memory: 6 GB of GDDR5 RAM

---

## 2. Using External Resources for More Details

If you want more detailed information about your GPU, you can use the following resources:

### **2.1 Wikipedia's NVIDIA GPU List**
- Wikipedia has an **exhaustive list** of all NVIDIA GPUs, even older models before CUDA.
- **What you’ll find:**
  - Memory type and capacity.
  - Compute capabilities.
  - GPU architecture and more.
- To find your GPU:
  1. Search for your GPU model (e.g., "Tesla C2075") in the Wikipedia page.
  2. Look at the highlighted information (e.g., **6 GB of GDDR5 RAM**).

### **2.2 TechPowerUp GPU Database**
- This is a user-friendly tool for finding GPU specifications.
- **What you’ll find:**
  - Memory type and size.
  - Bandwidth and memory bus.
  - Processor details and performance comparisons.
- Visit the [TechPowerUp GPU Database](https://www.techpowerup.com) and search for your GPU.

---

## 3. Practical Example: Tesla C2075
Let’s take the Tesla C2075 GPU as an example:
1. Run `nvidia-smi` and note the product name: Tesla C2075.
2. Search "Tesla C2075" on:
   - **Wikipedia**: Find that it has **6 GB of GDDR5 RAM**.
   - **TechPowerUp**: Get details about bandwidth, memory bus, and more.

---

## 4. Why is This Important?

- **Global Memory:** Knowing the memory helps you understand how much data your GPU can handle.
- **Additional Specs:** Details like memory bandwidth and bus width give insights into your GPU’s performance for specific workloads.

---

## 5. Summary
- Use `nvidia-smi` to quickly identify your GPU’s product name and memory stats.
- Use Wikipedia or TechPowerUp for detailed information.
- Understanding these details helps you optimize GPU usage for your projects.
