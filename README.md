# Spiking Neural Networks for Embedded Vision
## From image features to FPGA inference with Vitis HLS

This lab presents a practical workflow for training and deploying a spiking neural network (SNN) on an FPGA to classify images into four classes: Full Cable, Left End, Right End, and No Cable. Images are converted to grayscale, resized to 60 × 80 pixels, and normalized. Sobel filtering generates ON and OFF edge masks, which are averaged over 5 × 5 cells to produce 384 input features. A fully connected SNN with leaky integrate-and-fire (LIF) neurons is trained using PyTorch and snnTorch, with output membrane potentials used for classification. The trained weights and biases are quantized and exported for accelerator implementation and verification in Vitis HLS. The generated RTL is then integrated into a Zynq-based Vivado design. Finally, Python sends preprocessed features over UART to the FPGA board, and the returned predictions, inference times, and membrane scores are used to evaluate hardware performance.


## Requirements

- One of the following options for the Python workflow:
  - Ubuntu 24.04 LTS installed directly on your computer; or
  - Ubuntu 24.04 LTS running through Windows Subsystem for Linux 2 (WSL 2); or
  - Python installed directly on Windows (requires Windows-specific commands to create and activate the virtual environment; package installation commands remain the same).
- Python 3.12 with `venv` and `pip`
- A Python virtual environment with the following packages installed:
  - PyTorch (`torch==2.12.0`)
  - snnTorch (`snntorch==0.9.4`)
  - NumPy (`numpy==1.26.4`)
  - OpenCV (`opencv-python==4.10.0.84`)
  - Matplotlib (`matplotlib==3.10.9`)
  - scikit-learn (`scikit-learn==1.9.0`)
  - tqdm (`tqdm==4.67.3`)
- Sufficient RAM to cache the training and validation features
- AMD/Xilinx development tools: Vitis, Vitis HLS, and Vivado (course instructions use **version 2020.2**)
- PYNQ-Z2 FPGA board with Zynq-7020 (`xc7z020clg400-1`)
- USB Type-A to Micro-USB Type-B data cable for FPGA programming, UART communication, and power
- Course datasets and source files from the course GitHub repository

**Note:** Training can run on a CPU. GPU acceleration is optional and requires a CUDA-capable GPU, a compatible driver, and a compatible PyTorch installation. The script automatically selects CUDA when available; otherwise, it uses the CPU.

**Note:** The Python installation instructions below are provided for Linux (Ubuntu), including Ubuntu running under WSL 2. For native Windows, use Windows-specific commands to install Python and create and activate the virtual environment. The `python -m pip` commands remain the same once the environment is activated. Update the dataset and output paths in the source code to match your computer.

## Disclaimer

The FPGA deployment instructions were prepared using Vitis, Vitis HLS, and Vivado **2020.2**. Newer versions may have different menus, project workflows, or settings. Adjust the steps as needed for your installed software versions.

Installation of the FPGA development tools and access to the FPGA board require separate setup. These tools and the FPGA board are needed for hardware deployment, but are not required to train the SNN in Python.


