# SparkFish 3.2 Manual


<div align="center">
  <img src="./Logo.png" alt="SparkChess Logo" width="160" style="border-radius: 20%;">
  <h1>✨SparkFish 3.2✨</h1>
  <p>A lite chess engine with NNUE</p>
  <p>Supported:Android OEX /Windows</p>
  <div align="center">

A free and strong UCI chess engine powered by NNUE.Only 6 source code files.Fantastic Performance!

[![License](https://img.shields.io/github/license/Lester-chess/SparkFish-NNUE-Chess-Engine?style=flat-square&color=brightgreen)](LICENSE)
[![Latest Release](https://img.shields.io/github/v/release/Lester-chess/SparkFish-NNUE-Chess-Engine?style=flat-square&color=007ec6&label=OFFICIAL%20RELEASE)](https://github.com/Lester-chess/SparkFish-NNUE-Chess-Engine/releases/latest)
[![Commits Since](https://img.shields.io/github/commits-since/Lester-chess/SparkFish-NNUE-Chess-Engine/latest?style=flat-square&color=007ec6)](https://github.com/Lester-chess/SparkFish-NNUE-Chess-Engine/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/Lester-chess/SparkFish-NNUE-Chess-Engine?style=flat-square&color=blue&label=REPO%20SIZE)](https://github.com/Lester-chess/SparkFish-NNUE-Chess-Engine)

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![NNUE](https://img.shields.io/badge/NNUE-HalfKAv2-orange?style=flat-square)](https://github.com/Lester-chess/SparkFish-NNUE-Chess-Engine)
[![UCI Protocol](https://img.shields.io/badge/PROTOCOL-UCI-blueviolet?style=flat-square)](https://en.wikipedia.org/wiki/Universal_Chess_Interface)

</div>
  <!-- Windows 平台徽章 -->

  [![Platform - Windows](https://img.shields.io/badge/platform-Windows-0078D4?logo=windows&logoColor=white)](https://github.com/USERNAME/REPO/releases)

<!-- Android 平台徽章 -->
[![Platform - Android](https://img.shields.io/badge/platform-Android-34A853?logo=android&logoColor=white)](https://github.com/USERNAME/REPO/releases)
</div>

</div>

<!-- Lichess Classical -->
![Lichess Classical](https://img.shields.io/badge/dynamic/json?color=0080ff&label=Lichess%20Classical&logo=lichess&query=%24.perfs.classical.rating&url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSparkchessengine)

<!-- Lichess Rapid -->
![Lichess Rapid](https://img.shields.io/badge/dynamic/json?color=0080ff&label=Lichess%20Rapid&logo=lichess&query=%24.perfs.rapid.rating&url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSparkchessengine)

<!-- Lichess Blitz -->
![Lichess Blitz](https://img.shields.io/badge/dynamic/json?color=0080ff&label=Lichess%20Blitz&logo=lichess&query=%24.perfs.blitz.rating&url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSparkchessengine)

<!-- Lichess Bullet -->
![Lichess Bullet](https://img.shields.io/badge/dynamic/json?color=0080ff&label=Lichess%20Bullet&logo=lichess&query=%24.perfs.bullet.rating&url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSparkchessengine)
</div>

SparkFish is a high-performance chess engine compliant with the Universal Chess Interface (UCI) protocol, featuring lite HalfkAv2 NNUE eval.




Time control:Lichess Biltz/rapid/bullet

CCRL [blitz benchmark](https://www.computerchess.org/).

| Rank | Name | CCRL(Estimated) | Elo(Lichess) | + | - | games | score | oppo. | draws |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | [SparkFish 3.2.1](https://github.com/Lester-chess/SparkFish/releases/tag/v3.2.1) | 3295| 2717 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | [SparkFish 3.2.0](https://github.com/Lester-chess/SparkFish/releases/tag/v3.2.0) | 3250 | 2701 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | [SparkFish 3.1.0](https://github.com/Lester-chess/SparkFish/releases/tag/v3.1.0) | 3191 | 2690 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | [SparkFish 3.0.0](https://github.com/Lester-chess/SparkFish/releases/tag/v3.0) | 3055 | 2669 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | [SparkFish 1.4.1](https://github.com/Lester-chess/SparkFish/releases/tag/v1.4.1) | 2685 | 2491 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | [SparkFish 1.3.0](https://github.com/Lester-chess/SparkFish/releases/tag/v1.3) | 2585 | 2369 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | [SparkFish 1.0.0](https://github.com/Lester-chess/SparkFish/releases/tag/v1.0) | 2450 | 2350 | 0 | 0 | 0 | 0 | 0 | 0 |
---

## 1. System Requirements & Installation

### 1.1 Requirements
- **Linux** (Ubuntu 20.04+ recommended) or **Windows** (10/11 x86_64)
- **Compiler**: GCC / G++ (v9.0+ supporting C++17) or MSVC
- **Dependencies**: OpenMP (for multi-threaded tuning)
- **Compatibility Layer (Optional)**: Wine (required only if running Windows `.exe` on Linux instances)

### 1.2 Compilation from Source

#### On Linux (Native / AutoDL / VPS)
```bash
# Clone the repository
git clone https://github.com/your-repo/sparkchess.git
cd sparkchess

# Build using G++ with optimization flags
g++ -O3 -std=c++17 -fopenmp -march=native src/*.cpp -o bin/sparkchess

# Grant executable permissions
chmod +x bin/sparkchess
```

#### On Windows
- :Use the internal Bat to build the exe version 




---

## 2. Basic UCI Usage

SparkChess conforms to the standard UCI protocol. Launch the executable from a terminal or connect it to any chess GUI (Arena, Cutechess, Bankia, Fritz).

### 2.1 Interactive Terminal Mode
Start the engine directly:
```bash
./sparkchess
```

Common UCI commands:
```text
uci                       # Initialize engine and print engine parameters
isready                   # Check if engine is ready (replies 'readyok')
setoption name Threads value 8
setoption name Hash value 256
position startpos moves e2e4 e7e5
go depth 15               # Search to depth 15
go wtime 300000 btime 300000 winc 2000 binc 2000  # Timed search
quit                      # Terminate the engine
```

---

## 3.Pytorch NNUE Trainer Manual
# NNUE PyTorch Trainer User Guide

This document provides instructions on how to set up and use the NNUE PyTorch trainer for model training and exporting. Depending on your local environment setup preferences, the guide is divided into two scenarios: **Training without a Virtual Environment** and **Training with a Virtual Environment**.

## Scenario 1: Training without a Virtual Environment

If you are training directly in your system's global Python environment, please follow the steps below:

### 1. Environment Preparation & Dependency Installation

1. **Python Version Requirement**: **Python 3.11** is recommended.

2. **Install Basic Dependencies**: Open Command Prompt (CMD) in the root directory of the trainer and use `pip` to install the required dependencies listed in `requirements.txt`:

   ```cmd
   pip install -r requirements.txt
   ```

3. **Install PyTorch (CUDA Version)**: Based on your current Python version and GPU driver support, visit the [Official PyTorch Website](https://pytorch.org/get-started/locally/) to obtain and run the appropriate installation command for CUDA-accelerated PyTorch.

4. **Compile the Data Loader**: Navigate to the trainer folder and run the batch script to generate `training_data_loader.dll`:

   ```cmd
   build-dataloader-windows.bat
   ```

### 2. Dataset Preparation & Starting the Training

1. **Obtain the Dataset**:
   You can download `.binpack` dataset files from the official HuggingFace repository:

   * Dataset URL: [official-stockfish/master-binpacks](https://huggingface.co/datasets/official-stockfish/master-binpacks/tree/main)

   * **Training Recommendation**: It is recommended to use a dataset of around **10GB** and better to mix 4-5 binpacks and train for **30-35 epochs**.

2. **Run the Training Command**:
   Once the preparation is complete, run the following command in the command line to start training:

   ```cmd
   python train.py "1.binpack" "2.binpack" "3.binpack" "4.binpack" --features="HalfKAv2^" --l1=1024 --max-epochs=x
   ```

   > **Parameter Description**:
   >
   > * `"path/to/your_dataset.binpack"`: Replace this with the actual file path and name of your `.binpack` dataset.
   >
   > * `--max-epochs=x`: Replace `x` with your desired number of training epochs (e.g., `--max-epochs=25`).

### 3. Exporting (Serializing) the NNUE Model File

After training is complete, you need to convert the generated checkpoint file (`.ckpt`) into the `.nnue` format usable by the engine.

1. **Run the Serialization Command**:

   ```cmd
   python serialize.py "logs\training_logs\version_x\checkpoints\last.ckpt" "..\engine\custom_512x2.nnue" --features="HalfKAv2^" --l1=1024 --l2=16 --l3=32
   ```

   > **Note**: `version_x` in the path (e.g., `version_0`, `version_1`, etc.) varies depending on your training runs and parameters. Please check the `logs\training_logs` folder first to locate the directory containing the latest checkpoint file.

2. **Deployment & Application**:
   Ensure that the generated `.nnue` file and the engine binary file (`.exe`) are placed in the **same directory**. The engine will automatically load the network file upon startup.

## Scenario 2: Training with a Virtual Environment

If you prefer to keep your system environment clean by using a virtual environment (such as `venv` or `conda`) for training, please follow the workflow below:

### 1. Navigate to the Directory & Activate the Virtual Environment

1. Open Command Prompt (CMD) and navigate to the `nnue-pytorch` folder:

   ```cmd
   cd /d "path\to\your\nnue-pytorch"
   ```

2. **Activate the Virtual Environment** (choose the activation command based on the tool and location where you installed your virtual environment):

   * **If you are using `venv`**:

     ```cmd
     # Assuming the virtual environment folder is named venv and is in the current directory
     .\venv\Scripts\activate
     ```

   * **If you are using `conda`**:

     ```cmd
     conda activate your_env_name
     ```

### 2. Configuration & Training

Once the virtual environment is successfully activated (you will see `(your_env_name)` at the beginning of the command prompt), all subsequent steps are identical to **Scenario 1**:

1. Run `pip install -r requirements.txt` to install dependencies, and install the matching PyTorch CUDA version.

2. Run `build-dataloader-windows.bat` to generate `training_data_loader.dll`.

3. Use the `python train.py` command to load the dataset and start training.

4. Use the `python serialize.py` command to export the `.ckpt` file to a `.nnue` file and place it in the same directory as the engine binary.



## 4. Validation & Elo Testing (SPRT)

Validate that the updated weights produce a measurable Elo increase using Sequential Probability Ratio Testing (SPRT):

```bash
cutechess-cli \
  -engine name=SparkChess_Tuned cmd=./sparkchess-tuned \
  -engine name=SparkChess_Base  cmd=./sparkchess-base \
  -each proto=uci tc=10+0.1 \
  -rounds 2000 \
  -repeat \
  -openings file=openings.pgn format=pgn order=random \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -concurrency 10
```
---

# SparkFish 3.2 使用手册（中文版）

SparkFish 是一款兼容通用国际象棋接口（UCI）协议的高性能国际象棋引擎，搭载了轻量级的 HalfKAv2 NNUE 评估网络。

---

## 1. 系统要求与安装

### 1.1 系统要求
- **操作系统**：**Linux**（推荐 Ubuntu 20.04+）或 **Windows**（10/11 x86_64）
- **编译器**：GCC / G++（支持 C++17 的 v9.0+ 版本）或 MSVC
- **依赖项**：OpenMP（用于多线程调优）
- **兼容层（可选）**：Wine（仅在 Linux 实例上运行 Windows `.exe` 文件时需要）

### 1.2 从源码编译

#### 在 Linux 上（原生 / AutoDL / VPS）
```bash
# 克隆代码仓库
git clone https://github.com/your-repo/sparkchess.git
cd sparkchess

# 使用带有优化标志的 G++ 进行编译
g++ -O3 -std=c++17 -fopenmp -march=native src/*.cpp -o bin/sparkchess

# 赋予可执行权限
chmod +x bin/sparkchess
```

#### 在 Windows 上
- 使用内置的批处理脚本（Bat）来构建 exe 版本。

---

## 2. UCI 基础用法

SparkFish 遵循标准的 UCI 协议。您可以直接从终端启动可执行文件，或将其连接到任何国际象棋图形界面（如 Arena、Cutechess、Bankia、Fritz）。

### 2.1 交互式终端模式
直接启动引擎：
```bash
./sparkchess
```

常用 UCI 命令：
```text
uci                       # 初始化引擎并打印引擎参数
isready                   # 检查引擎是否就绪（返回 'readyok'）
setoption name Threads value 8
setoption name Hash value 256
position startpos moves e2e4 e7e5
go depth 15               # 搜索至 15 层深度
go wtime 300000 btime 300000 winc 2000 binc 2000  # 限时搜索
quit                      # 终止并退出引擎
```

---

## 3. PyTorch NNUE 训练器手册
# NNUE PyTorch 训练器使用指南

本文档旨在介绍如何配置并使用 NNUE PyTorch 训练器进行模型训练与导出。根据您的本地环境配置习惯，分为**不使用虚拟环境**和**使用虚拟环境**两种情况进行说明。

## 情况一：不使用虚拟环境进行训练

如果您直接在系统的全局 Python 环境中进行训练，请按照以下步骤操作：

### 1. 环境准备与依赖安装

1. **Python 版本要求**：推荐使用 **Python 3.11**。

2. **安装基础依赖**：在训练器根目录下打开命令提示符（CMD），使用 `pip` 安装 `requirements.txt` 中要求的功能组件：

   ```cmd
   pip install -r requirements.txt
   ```

3. **安装 PyTorch (CUDA 版本)**：根据您当前使用的 Python 版本和显卡驱动支持情况，前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取并运行合适的带有 CUDA 加速的 PyTorch 安装命令。

4. **编译数据加载器**：进入训练器文件夹内，运行批处理文件以生成 `training_data_loader.dll`：

   ```cmd
   build-dataloader-windows.bat
   ```

### 2. 数据集准备与开始训练

1. **获取数据集**：
   您可以前往 HuggingFace 官方仓库下载 `.binpack` 格式的数据集文件：

   * 数据集地址：[official-stockfish/master-binpacks](https://huggingface.co/datasets/official-stockfish/master-binpacks/tree/main)

   * **训练建议**：推荐使用 **10GB 左右** 的数据集，可多个混合训练 **30 - 35 个轮次（Epochs）**。

2. **执行训练命令**：
   准备工作结束后，在命令行中使用以下命令开始训练：

   ```cmd
  python train.py "1.binpack" "2.binpack" "3.binpack" "4.binpack" --features="HalfKAv2^" --l1=1024 --max-epochs=x
   ```

   > **参数说明**：
   >
   > * `"你的binpack文件名称和位置.binpack"`：请替换为您实际的数据集文件路径及名称。
   >
   > * `--max-epochs=x`：将 `x` 替换为您自定义的训练轮次数（例如 `--max-epochs=25`）。

### 3. 导出（序列化）NNUE 模型文件

训练结束后，需要将生成的检查点文件（`.ckpt`）转换为引擎可用的 `.nnue` 格式。

1. **执行序列化命令**：

   ```cmd
   python serialize.py "logs\training_logs\version_x\checkpoints\last.ckpt" "..\engine\custom_512x2.nnue" --features="HalfKAv2^" --l1=1024 --l2=16 --l3=32
   ```

   > **注意**：路径中的 `version_x`（如 `version_0`、`version_1` 等）根据您的训练次数和参数情况会有所不同。请先在 `logs\training_logs` 文件夹中查看，并定位最新生成的检查点文件所在目录。

2. **部署与应用**：
   将生成后的 `.nnue` 文件与引擎二进制可执行文件（`.exe`）存放在**同一目录下**，引擎启动时会自动读取该网络文件。

## 情况二：使用虚拟环境进行训练

如果您希望保持系统环境整洁，使用虚拟环境（如 `venv` 或 `conda`）进行训练，请按照以下流程操作：

### 1. 进入目录并激活虚拟环境

1. 打开命令提示符（CMD），首先进入 `nnue-pytorch` 文件夹：

   ```cmd
   cd /d "您的nnue-pytorch文件夹路径"
   ```

2. **激活虚拟环境**（具体命令根据您安装虚拟环境的工具和位置来决定）：

   * **如果您使用的是 `venv`**：

     ```cmd
     # 假设虚拟环境文件夹名为 venv 且在当前目录下
     .\venv\Scripts\activate
     ```

   * **如果您使用的是 `conda`**：

     ```cmd
     conda activate 您的虚拟环境名称
     ```

### 2. 配置与训练

虚拟环境激活成功后（命令行前端会出现 `(环境名称)` 提示），后续所有步骤与**情况一**完全一致：

1. 运行 `pip install -r requirements.txt` 安装依赖，并安装匹配的 PyTorch CUDA 版本。

2. 运行 `build-dataloader-windows.bat` 生成 `training_data_loader.dll`。

3. 使用 `python train.py` 命令加载数据集开始训练。

4. 使用 `python serialize.py` 命令将 `.ckpt` 导出为 `.nnue` 文件，并与引擎二进制文件放在同一目录下。

---

## 4. 验证与 Elo 测试 (SPRT)

使用序贯概率比检验（SPRT）来验证更新后的权重是否能带来可观的 Elo 等级分提升：

```bash
cutechess-cli \
  -engine name=SparkChess_Tuned cmd=./sparkchess-tuned \
  -engine name=SparkChess_Base  cmd=./sparkchess-base \
  -each proto=uci tc=10+0.1 \
  -rounds 2000 \
  -repeat \
  -openings file=openings.pgn format=pgn order=random \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -concurrency 10
```
