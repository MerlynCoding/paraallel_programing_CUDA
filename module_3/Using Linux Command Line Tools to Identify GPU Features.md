
# Using Linux Command Line Tools to Identify GPU Features

This README will help you learn how to use simple Linux command-line tools to check specific features of your NVIDIA GPU, including global memory, bandwidth, and more.

---

## 1. Key Tools for GPU Information

### **1.1 `nvidia-smi` Command**
The `nvidia-smi` command is the primary tool for getting detailed information about your NVIDIA GPUs. Without any arguments, it will display a summary of all NVIDIA GPUs in your system.

#### Information provided by `nvidia-smi`:
- **Memory capabilities**: Amount and usage of GPU memory.
- **Clock speeds**: Speeds for the graphics interface.
- **Processor details**: Streaming multiprocessor (SM) information.
- **Temperature**: Current GPU temperature.
- **Utilization**: How much the GPU is being used.
- **Many more details**: Power consumption, fan speed, etc.

#### Common `nvidia-smi` Options:
- **`-L` or `--list-gpus`**: Lists all GPUs and their UUIDs (unique identifiers).
- **`-i [GPU ID]`**: Shows information for a specific GPU.
- **`-d [info type]`**: Displays specific stats (e.g., memory, utilization, clock speed).

---

### **1.2 `lspci` Command**
The `lspci` command lists all PCI (Peripheral Component Interconnect) devices connected to your machine, including GPUs.

#### How to Use `lspci`:
- Run `lspci` to see all PCI devices.
- Use the `grep` command with "NVIDIA" to show only NVIDIA devices:
  ```bash
  lspci | grep NVIDIA
  ```
- Adjust verbosity levels with the `-v` option for more details.
- Add `--color` to make the output easier to read.

> **Note**: The `lspci` command may not be available on all systems. If not, install it using your Linux package manager.

---

## 2. Practical Example: Checking GPU Memory
Here’s a quick example to get the memory details of your NVIDIA GPU:

1. Open a terminal.
2. Run the command:
   ```bash
   nvidia-smi -d memory
   ```
3. The output will show:
   - Total memory.
   - Used memory.
   - Free memory.

---

## 3. Tips for Beginners
- **Start with `nvidia-smi`:** It's the easiest and most reliable tool for NVIDIA GPUs.
- **Use `lspci` for hardware checks:** This helps confirm that your GPU is connected and recognized.
- **Experiment with options:** Try different `nvidia-smi` arguments (like `-L` or `-i`) to explore more about your GPU.
