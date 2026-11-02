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
  - pyserial (`pyserial==3.5`)
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


## Instructions

**Note:** Some packages and versions differ from those used in the CNN lab. Create a separate virtual environment named `snn_env` for this lab.

### 1. Install Python prerequisites

Open an Ubuntu terminal. If you are using WSL 2 on Windows, open your Ubuntu terminal.

Install Python 3.12, `venv`, and `pip`:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

### 2. Create and activate a Python virtual environment

Open a terminal in your project folder and create a virtual environment named `snn_env` using Python 3.12:

```bash
/usr/bin/python3.12 -m venv snn_env
```

Activate the virtual environment:

```bash
source snn_env/bin/activate
```

Verify the Python version:

```bash
python --version
```

The output should show `Python 3.12.x`. Keep the virtual environment activated when installing packages or running the Python scripts.

### 3. Install the required Python packages

With the `snn_env` virtual environment activated, upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the required packages using the specified versions:

```bash
python -m pip install torch==2.12.0 snntorch==0.9.4 numpy==1.26.4 opencv-python==4.10.0.84 matplotlib==3.10.9 scikit-learn==1.9.0 tqdm==4.67.3 pyserial==3.5
```

The packages are used for:

- **PyTorch:** building and training the neural network, computing gradients, and saving model weights
- **snnTorch:** implementing leaky integrate-and-fire neurons and surrogate gradients for SNN training
- **NumPy:** numerical arrays, feature processing, and weight quantization
- **OpenCV (`opencv-python`):** image loading, resizing, and Sobel edge extraction
- **Matplotlib:** plotting ON/OFF spike maps and training and validation accuracy
- **scikit-learn:** splitting the dataset into training and validation sets while preserving class proportions
- **tqdm:** displaying progress bars during feature caching
- **pyserial:** sending quantized image features to the FPGA and receiving predictions, execution times, and logits over UART

**Note:** CPU training is supported. GPU acceleration is optional and requires a CUDA-capable GPU, a compatible driver, and a compatible PyTorch installation. The training script automatically selects CUDA when available; otherwise, it uses the CPU.

### 4. Verify the installed packages

List the installed packages and their versions:

```bash
python -m pip list
```

Check for dependency conflicts:

```bash
python -m pip check
```

Verify that all required packages can be imported:

```bash
python -c "import torch, snntorch, numpy, cv2, matplotlib, sklearn, tqdm, serial; print('All required packages imported successfully.'); print('CUDA available:', torch.cuda.is_available())"
```

### 5. Install Vitis, Vitis HLS, and Vivado

Download the **Unified Installer for version 2020.2** from the [AMD download page](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2020-2.html).

Run the installer and select **Vitis** as the installation product. Ensure that **Vitis, Vitis HLS, and Vivado** are included, along with device support for **Zynq-7000**.

Complete the installation. **Version 2020.2 is preferred** because it matches the tools used in this lab.

**Version note:** You may use a newer release, but menus, tool names, and project workflows may differ.

**Note:** If these tools and Zynq-7000 device support are already installed from the CNN lab, you do not need to reinstall them. The FPGA tools are required for hardware deployment, but are not needed for Python training.



