# AdaptiveCore
This is the code of our paper "AdaptiveCore: Adaptive Parallel Core Decomposition Framework" submitted to IEEE Transactions on Computers. AdaptiveCore is a high-performance GPU framework for k-core decomposition on large graphs.

- Our paper is currently under revision.

## 🔧 Requirements

- **CUDA Compiler**: `nvcc` (version ≥ 12)
- **C++ Compiler**: `g++` (version ≥ 14)

## 🛠️ Build Instructions

```bash
cd ./AdaptiveCore
make
```

## 🚀 Execution
```bash
./adaptivecore <input_file> <device_id> <repeat_times>
```
- Example:
```bash
./adaptivecore ./datasets/01_amazon0601-symmetric.mtx 0 10
```
This example runs the k-core algorithm on the Amazon dataset using GPU device 0, repeating 10 times for performance measurement.

## 📄 Input Graph Format
The default input is a CSR-formatted graph stored as a plain text file with the following structure:
```bash
<n>     // Number of vertices
<m>     // Number of edges

// Degree list (n lines)
d0
d1
...
d(n-1)

// Offset list (n lines)
o0
o1
...
o(n-1)

// Edge list (m lines)
e0
e1
...
e(m-1)
```

### 📎 Alternative: Matrix Market (.mtx) Format
You may also load .mtx format graphs (e.g., from SuiteSparse or SNAP), but this requires modifying the source code.

### 📌 Notes
Ensure the input graph without self-loops is undirected and symmetric for correct k-core results.

Multiple repetitions can be used to measure GPU performance stability.

### 🧪 Sample Datasets
You can find test datasets in ./datasets/, or download from:
- SNAP
- SuiteSparse Matrix Collection

## 📬 Contact
If you encounter any issues or want to contribute, feel free to open an issue or contact the author.

## 📝 Related Papers

Please consider citing our paper. Thank you!
<!--
```bash
@inproceedings{SpeedCore,
author = {Zhao, Chen and Yu, Ting and Zheng, Zhigao and Zhu, Yuanyuan and Jin, Song and Du, Bo and Tao, Dacheng},
title = {SpeedCore: Space-efficient and Dependency-aware GPU Parallel Framework for Core Decomposition},
year = {2024},
url = {https://doi.org/10.1145/3673038.3673111},
doi = {10.1145/3673038.3673111},
booktitle = {Proceedings of the 53rd International Conference on Parallel Processing},
pages = {555–564},
numpages = {10},
location = {Gotland, Sweden},
series = {ICPP '24}
}
```
-->
