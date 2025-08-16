[![author](https://img.shields.io/badge/author-mohd--faizy-red)](https://github.com/mohd-faizy)
![made-with-Markdown](https://img.shields.io/badge/Made%20with-markdown-blue)
![Language](https://img.shields.io/github/languages/top/mohd-faizy/learn_python)
![Maintained](https://img.shields.io/maintenance/yes/2025)
![Last Commit](https://img.shields.io/github/last-commit/mohd-faizy/PyTorch-Essentials)
[![contributions welcome](https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=flat-square)](https://github.com/mohd-faizy/PyTorch-Essentials)
![Size](https://img.shields.io/github/repo-size/mohd-faizy/PyTorch-Essentials)

# PyTorch Essentials

![PyTorch](https://github.com/mohd-faizy/PyTorch-Essentials/blob/main/_img/pytorch.jpg?raw=true)

**Welcome to PyTorch Essentials, a comprehensive repository covering the power and versatility of PyTorch, a cutting-edge deep learning library.**

## Table of Contents

- [PyTorch Essentials](#pytorch-essentials)
  - [Table of Contents](#table-of-contents)
  - [🚀 Why PyTorch?](#-why-pytorch)
    - [Understanding Tensors in PyTorch](#understanding-tensors-in-pytorch)
    - [🗄️**Data Types**](#️data-types)
    - [**📅 PyTorch Release Timeline**](#-pytorch-release-timeline)
      - [**PyTorch 0.1 (2017)**](#pytorch-01-2017)
      - [**PyTorch 1.0 (2018)**](#pytorch-10-2018)
      - [**PyTorch 1.x Series**](#pytorch-1x-series)
      - [**PyTorch 2.0**](#pytorch-20)
  - [**🕸️ History of PyTorch**](#️-history-of-pytorch)
      - [**The Origins**](#the-origins)
      - [**The Rise of PyTorch (`Python` + `Torch`)**](#the-rise-of-pytorch-python--torch)
      - [**Milestones in PyTorch’s Development**](#milestones-in-pytorchs-development)
      - [**Impact and Adoption**](#impact-and-adoption)
    - [**PyTorch Release Timeline**](#pytorch-release-timeline)
  - [**⚡ Challenges and Solutions**](#-challenges-and-solutions)
      - [**Pre-PyTorch Challenges**](#pre-pytorch-challenges)
      - [**PyTorch's Solutions**](#pytorchs-solutions)
  - [**🔄 Core Features**](#-core-features)
      - [1. **⚙️ Dynamic Computation Graphs**](#1-️-dynamic-computation-graphs)
      - [2. **📈 Autograd**](#2--autograd)
      - [3. **🚀 GPU Acceleration**](#3--gpu-acceleration)
      - [4. **📚 TorchScript**](#4--torchscript)
      - [5. **📊 ONNX Compatibility**](#5--onnx-compatibility)
      - [6. **🔧 Distributed Training**](#6--distributed-training)
  - [**🔧 PyTorch vs TensorFlow**](#-pytorch-vs-tensorflow)
  - [**🔄 Core Modules**](#-core-modules)
  - [**📊 Domain Libraries**](#-domain-libraries)
  - [**💡 Popular Ecosystem Libraries**](#-popular-ecosystem-libraries)
  - [🛣️Roadmap](#️roadmap)
  - [📒Colab-Notebook-1](#colab-notebook-1)
  - [📒Colab-Notebook-2](#colab-notebook-2)
  - [🔦Explore](#explore)
  - [💧PyTorch code](#pytorch-code)
  - [⚡PyTorch APIs](#pytorch-apis)
  - [⭐Who uses PyTorch](#who-uses-pytorch)
  - [🍰 Contributing](#-contributing)
  - [🙇 Acknowledgements](#-acknowledgements)
  - [⚖ ➤ License](#--license)
  - [❤️ Support](#️-support)
  - [🔗Connect with me:](#connect-with-me)

## 🚀 Why PyTorch?

- **Open-Source Deep Learning Library**: Developed by **Meta AI** (formerly **Facebook AI Research**).
- **Python & Torch**: Combines Python’s ease of use with the efficiency of the Torch scientific computing framework, originally built with **Lua**. Torch was known for high-performance tensor-based operations, especially on GPUs.

### Understanding Tensors in PyTorch

*Tensors are multi-dimensional arrays and the core data structure in PyTorch.*
![Tensor](https://raw.githubusercontent.com/mohd-faizy/PyTorch-Essentials/refs/heads/main/_img/Tensor.png)

### 🗄️**Data Types**

| **Data Types**         | **Dtype**         | **Description**                                                                                                                                       |
|------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **32-bit Floating Point** | `torch.float32`   | Standard floating-point type used for most deep learning tasks. Provides a balance between precision and memory usage.                            |
| **64-bit Floating Point** | `torch.float64`   | Double-precision floating point. Useful for high-precision numerical tasks but uses more memory.                                                  |
| **16-bit Floating Point** | `torch.float16`   | Half-precision floating point. Commonly used in mixed-precision training to reduce memory and computation.                                        |
| **BFloat16**              | `torch.bfloat16`  | Brain floating-point format with reduced precision compared to `float16`. Used in mixed-precision training.                                       |
| **8-bit Floating Point**  | `torch.float8`    | Ultra-low-precision floating point. Used for experimental applications and extreme memory-constrained scenarios.                                  |
| **8-bit Integer**         | `torch.int8`      | 8-bit signed integer. Used for quantized models to save memory and computation in inference.                                                      |
| **16-bit Integer**        | `torch.int16`     | 16-bit signed integer. Useful for special numerical tasks requiring intermediate precision.                                                       |
| **32-bit Integer**        | `torch.int32`     | Standard signed integer type. Commonly used for indexing and general-purpose numerical tasks.                                                     |
| **64-bit Integer (Long Tensor)** | `torch.int64`     | Long integer type. Often used for large indexing arrays or tasks involving large numbers.                                                  |
| **8-bit Unsigned Integer** | `torch.uint8`     | 8-bit unsigned integer. Commonly used for image data (e.g., pixel values between 0 and 255).                                                     |
| **Boolean**             | `torch.bool`      | Boolean type. Stores `True` or `False` values. Often used for masks in logical operations.                                                          |
| **Complex 64**          | `torch.complex64` | Complex number type with 32-bit real and 32-bit imaginary parts. Used for scientific and signal processing tasks.                                   |
| **Complex 128**         | `torch.complex128`| Complex number type with 64-bit real and 64-bit imaginary parts. Offers higher precision but uses more memory.                                      |
| **Quantized Integer**   | `torch.qint8`     | Quantized signed 8-bit integer. Used in quantized models for efficient inference.                                                                   |
| **Quantized Unsigned Integer** | `torch.quint8`    | Quantized unsigned 8-bit integer. Often used for quantized tensors in image-related tasks.                                                   |

### **📅 PyTorch Release Timeline**

#### **PyTorch 0.1 (2017)**

- **Key Features**:
  - Introduced the **dynamic computation graph**, enabling more flexible model architectures.
  - Seamless integration with other Python libraries (e.g., **NumPy**, **SciPy**).
- **Impact**:
  - Gained popularity among researchers due to its intuitive, Pythonic interface and flexibility.
  - Quickly featured in numerous research papers.

#### **PyTorch 1.0 (2018)**

- **Key Features**:
  - Bridged the gap between **research** and **production** environments.
  - Introduced **TorchScript** for model serialization and optimization.
  - Improved performance with **Caffe2** integration.
- **Impact**:
  - Enabled smoother transitions of models from **research to deployment**.

#### **PyTorch 1.x Series**

- **Key Features**:
  - Support for **distributed training**.
  - **ONNX** compatibility for interoperability with other frameworks.
  - Introduced **quantization** for model compression and efficiency.
  - Expanded ecosystem with **torchvision** (CV), **torchtext** (NLP), and **torchaudio** (audio).
- **Impact**:
  - Increased adoption by the **research community** and **industry**.
  - Inspired community libraries like **PyTorch Lightning** and **Hugging Face Transformers**.
  - Strengthened **cloud support** for easy deployment.

#### **PyTorch 2.0**

- **Key Features**:
  - Significant **performance improvements**.
  - Enhanced support for **deployment** and **production-readiness**.
  - Optimized for modern hardware (**TPUs**, custom **AI chips**).
- **Impact**:
  - Improved speed and scalability for **real-world applications**.
  - Better compatibility with a variety of **deployment environments**.

## **🕸️ History of PyTorch**

#### **The Origins**

- 🧰 **Torch**: Developed by the **Idiap Research Institute** (2002), written in **Lua** + **C**.
- ⚙️ **Flexible Tensors**: Introduced an N-dimensional array (Tensor) for scientific computing and ML.
- 🚧 **Challenges**:
  - **Lua Ecosystem**: Limited community adoption in the Python-dominant ML space.
  - **Static Computation Graphs**: Hard to work with dynamic tasks like sequence modeling.
  - 🔄 **Research to Production Gap**: Lacked seamless production integration.

#### **The Rise of PyTorch (`Python` + `Torch`)**

- **PyTorch**: Released by **Meta AI** in **September 2016** as the Python-friendly evolution of Torch.
- 🔧 **Key Improvements**:
  - **Dynamic Computation Graphs**: Allowed real-time modification and debugging.
  - **Pythonic API**: Seamless integration with **NumPy**, **SciPy**, and other Python libraries.
  - 🚀 **Seamless GPU Support**: Optimized for **NVIDIA CUDA**, boosting performance.
  - 🔍 **Enhanced Debugging**: Dynamic nature enabled intuitive debugging with Python tools.

#### **Milestones in PyTorch’s Development**

- 📅 **2018**: Integrated with **Caffe2** to enhance deployment and scalability.
- 🌐 **2022**: **PyTorch Foundation** was established under **Linux Foundation** to ensure open governance.

#### **Impact and Adoption**

- 🌍 **Widely Used** in **CV**, **NLP**, and **AI** across academia and industry.
- 🌱 **Community-Driven**: Continuous growth through contributions, research papers, and open-source collaborations.

* * *

### **PyTorch Release Timeline**

| **Version** | **Release Date** | **Key Features** |
| --- | --- | --- |
| 0.1 | 🌐 January 2017 | Dynamic graphs and CUDA support. |
| 0.2 | ⏳ June 2017 | Expanded tensor operations. |
| 1.0 | ✨ December 2018 | Caffe2 integration; TorchScript. |
| 1.3 | 🌐 October 2019 | ONNX runtime; quantization support. |
| 1.5 | 🌟 May 2020 | TorchServe for production. |
| 1.9 | 🔄 June 2021 | Profiler, M1 GPU support. |
| 2.0 | 🔥 March 2023 | Optimized dynamic graph via TorchDynamo. |

## **⚡ Challenges and Solutions**

#### **Pre-PyTorch Challenges**

1. **Static Graphs**:

    - 🔒 Rigid structures, difficult for dynamic tasks.
    - 🚧 Limited debugging and experimentation.
2. **Research to Production**:

    - 🚫 Poor transition between experimental models and real-world deployment.
3. **Ecosystem Fragmentation**:

    - 🔄 Lack of Pythonic support; Lua and TensorFlow had steep learning curves.
4. **Performance Bottlenecks**:

    - ⚡ GPU acceleration was often convoluted.

#### **PyTorch's Solutions**

1. **🌐 Dynamic Graphs**:

    - Created at runtime, enabling flexibility.
    - 💡 **Example**: Sequence models (e.g., RNNs) work seamlessly.
2. **📚 Pythonic API**:

    - Simplified coding and debugging. Integrated natively with **NumPy** and **SciPy**.
3. **🎁 Unified Research + Production**:

    - 🔄 **TorchScript** & **ONNX** bridge research models into production.
4. **🚀 Seamless GPU Acceleration**:

    - Built-in **CUDA** support for fast deep learning.
5. **🤝 Community-Driven**:

    - Active participation from academia, startups, and research labs.

## **🔄 Core Features**

#### 1. **⚙️ Dynamic Computation Graphs**

- Real-time graph construction for dynamic models.
- 💻 **Advantage**: Easier to experiment with custom layers and operations.

#### 2. **📈 Autograd**

- Automatic differentiation engine for easy backpropagation.
- 🌟 **Use Case**: Critical for training deep networks.

#### 3. **🚀 GPU Acceleration**

- Native **CUDA** support enables fast training.
- 🖥️ **Optimized for NVIDIA GPUs**.

#### 4. **📚 TorchScript**

- Serializes PyTorch models for production, with optimizations.
- 🔧 **Deployable**: Easily port models to other languages or systems.

#### 5. **📊 ONNX Compatibility**

- Export PyTorch models to ONNX format for multi-framework interoperability.
- 🚀 **Deployment**: Use ONNX in TensorFlow, MXNet, or other frameworks.

#### 6. **🔧 Distributed Training**

- Multi-GPU and multi-node training using `torch.distributed`.
- ⚡ **Scalable**: Efficiently handles large models and datasets.

## **🔧 PyTorch vs TensorFlow**

| **Aspect**                         | **PyTorch**                              | **TensorFlow (Pre-2.x)**                 | **Verdict**                                               |
|------------------------------------|------------------------------------------|-----------------------------------------|-----------------------------------------------------------|
| **Programming Language**           | Pythonic (Python-first design)           | Supports Mulitple languages: Python, C++, Java, Tensorflow.js, Swift.  | PyTorch has a more Pythonic, user-friendly design.        |
| **Ease of Use**                    | Intuitive, easier to learn               | TensorFlow `2.x` improved usability with keras integration, but can be still complex.  | PyTorch is easier to use, especially for beginners.       |
| **Deployment and Production**      | TorchScript, ONNX                        | TensorFlow Lite, TensorFlow Extended (TFX), TensorFlow.js | TensorFlow has more robust production deployment tools.  |
| **Performance**                    | Good, especially on GPUs, Offer TorchScript for model serialization   | Highly optimized, very fast, use XLA compiler             | TensorFlow is often more optimized for production workloads. |
| **Community and Ecosystem**        | Fast-growing, research-centric, smaller ecosystem | Established, larger ecosystem, but fragmented | TensorFlow has a larger, more mature community.           |
| **High-Level API**                 | Higher-level APIs (e.g., `torch.nn`), PyTorch Lightning and Fast.ai     | Keras (integrated in TensorFlow 2.x)    | TensorFlow’s Keras API is more mature but less flexible.  |
| **Mobile and Embedded Deployment** | Limited support (via TorchScript, LibTorch) | TensorFlow Lite                         | TensorFlow Lite has more support for mobile and embedded devices. |
| **Preferred Domain**               | Research, academia, prototyping          | Production, web, mobile, large-scale systems | TensorFlow is preferred for production-ready systems, while PyTorch excels in research. |
| **Learning Curve**                 | Moderate (easy to get started)           | Steep (especially in pre-2.x)            | PyTorch has a gentler learning curve.                    |
| **Interoperability**               | Supports ONNX, Python-first ecosystem    | Limited interoperability (Pre-2.x), Now use TensorFlow Hub and Saved Model; Support ONNX with some limitations.      | PyTorch is better in terms of interoperability with other tools. |
| **Customizability**                | High (flexible design, dynamic graphs)   | Moderate (requires more effort for custom solutions) | PyTorch is more customizable due to its dynamic nature.   |
| **Deployment Tools**               | TorchServe, ONNX, cloud support          | TensorFlow Serving, TensorFlow Extended (TFX) for ML pipelines.                  | TensorFlow has more production deployment tools and frameworks. |
| **Parallelism and Distributed Training** | Distributed data parallelism, `torch.distributed` | Extensive Support with `tf.distribute.Strategy`; optimized for large-scale computing | TensorFlow is generally more mature for distributed training. |
| **Model Zoo and Pretrained Models** | Access via `TorchVision` Hugging Face; strong community sharing | TensorFlow Hub offer a wide range; extensive community models   | Both offer extensive Pre-trained models |

## **🔄 Core Modules**

| **Module**                    | **Description**                                                                 | **Key Features/Functions**                                   |
|-------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------|
| **`torch`**                      | The base PyTorch library that includes core functions and tensor operations.    | Tensor operations, device management, and random number generation. |
| **`torch.autograd`**             | Automatic differentiation for computing gradients.                             | `autograd.Function`, `backward()`, and gradient computation. |
| **`torch.nn`**                   | Module for building neural networks with layers, losses, and optimizers.        | Layers (e.g., `nn.Linear`, `nn.Conv2d`), loss functions (e.g., `nn.CrossEntropyLoss`). |
| **`torch.optim`**                | Optimization algorithms like SGD, Adam, and more.                               | Optimizers like `SGD`, `Adam`, and `RMSprop`.               |
| **`torch.utils.data`**           | Utility functions for data handling, including Dataset and DataLoader classes.  | `Dataset`, `DataLoader`, and batch processing.              |
| **`torch.jit`**                  | JIT (Just-In-Time) compiler for optimizing models.                              | Model serialization with `torch.jit.script`, speed optimization. |
| **`torch.distributed`**          | Distributed computing for training across multiple devices or nodes.            | Multi-GPU training, `DistributedDataParallel`.               |
| **`torch.cuda`**                 | CUDA interface for GPU acceleration.                                            | GPU tensor operations (`cuda()`, `to(device)`, etc.).       |
| **`torch.backend`**              | Backend operations for execution on different hardware.                         | Execution backends like XLA (for TPUs), and others.         |
| **`torch.multiprocessing`**      | For parallel computing with multiple processes.                                | Multi-process data loading and execution.                   |
| **`torch.quantization`**         | Quantization tools for reducing model size and improving inference speed.       | Model conversion to `qint8` for faster deployment.          |
| **`torch.onnx`**                 | Interoperability with other frameworks through the Open Neural Network Exchange (ONNX) format. | Export models to ONNX with `torch.onnx.export`.             |

## **📊 Domain Libraries**

| **Library** | **Description** | **Key Features/Functions** |
| --- | --- | --- |
| **`torchvision`** | Computer vision domain library for tasks like image classification, detection, etc. | Pre-trained models (e.g., ResNet, VGG), transforms (e.g., `Resize`, `ToTensor`), dataset classes (`CIFAR10`, `ImageNet`). |
| **`torchaudio`** | Library for audio processing tasks, including speech recognition, sound classification, and more. | Audio loading (`torchaudio.load`), spectrograms, pre-trained models for speech recognition. |
| **`torchtext`** | Natural language processing (NLP) domain library for text-based tasks. | Text preprocessing, tokenization, embeddings (e.g., GloVe), dataset classes (`IMDB`, `TextClassification`). |
| **`torchmetrics`** | Metrics computation library for deep learning tasks. | Metric classes for accuracy, precision, recall, F1 score, etc. for various tasks (e.g., classification, regression). |
| **`torchgeo`** | PyTorch library for geospatial and remote sensing data. | Geo-specific datasets, models, and transforms for satellite imagery and geospatial data processing. |
| **`torchrl`** | Reinforcement learning library for building and training RL models. | RL-specific models (e.g., DQN, A3C), environments, and training pipelines. |
| **`torchquantum`** | Quantum machine learning library built on top of PyTorch. | Quantum circuits, quantum data processing, hybrid classical-quantum learning. |
| **`torchdrug`** | Deep learning library for computational biology and drug discovery. | Molecular graphs, cheminformatics, and graph neural networks for drug discovery. |
| **`torchbio`** | Bioinformatics-focused library for deep learning with biological datasets. | Sequence models, genomic data processing, biological datasets. |
| **`torchrec`** | PyTorch library for building and deploying recommendation systems. | Embedding models, ranking algorithms, and efficient data handling for recommendation tasks. |
| **`torcharrow`** | Library for accelerated data loading and preprocessing, especially for tabular and time series data. | High-performance data processing, optimized for ML workflows (experimental). |
| **`torchserve`** | Model serving library for deploying PyTorch models at scale in production. | REST APIs, scalable serving, model versioning, inference logging. |
| **`pytorch_lightning`** | High-level wrapper that simplifies training loops and reduces boilerplate. | Trainer class, distributed training support, reproducible and scalable model training. |


## **💡 Popular Ecosystem Libraries**

| **Library** | **Description** | **Key Features/Functions** |
| --- | --- | --- |
| **Hugging Face Transformers** | State-of-the-art NLP models and tools for text processing. | Pre-trained models (BERT, GPT-2, T5, etc.), tokenization, fine-tuning for NLP tasks. |
| **fastai** | High-level library built on top of PyTorch, designed for ease of use in deep learning. | Simplified API for training neural networks, transfer learning, and pre-trained models. |
| **PyTorch Geometric** | Library for deep learning on graphs and geometric data. | Graph neural networks (GNNs), graph datasets, and data processing for graph-based tasks. |
| **TorchMetrics** | Metrics library for monitoring performance across a wide range of machine learning tasks. | Pre-built metrics like accuracy, F1-score, confusion matrix, etc. for classification, regression, etc. |
| **TorchElastic** | Scalable elastic deep learning on Kubernetes. | Job management, checkpointing, fault-tolerant training, and scaling distributed training. |
| **Optuna** | Hyperparameter optimization framework. | Automatic hyperparameter search, optimization algorithms (TPE, CMA-ES), integration with PyTorch models. |
| **Catalyst** | High-level framework for training deep learning models with minimal code. | Experiment management, model training, and evaluation with support for distributed training. |
| **Ignite** | High-level library for training neural networks, simplifying deep learning workflows. | Simplifies training loops, metrics tracking, and provides extensibility with events and handlers. |
| **AllenNLP** | NLP-focused library built on top of PyTorch for research and production. | Models and tools for text-based tasks (e.g., question answering, text classification). |
| **Skorch** | Sklearn-like interface for training PyTorch models. | Integration of PyTorch models with Scikit-learn API for easier experimentation and hyperparameter tuning. |
| **PyTorch Forecasting** | Time series forecasting library built on top of PyTorch. | Pre-built models for time series forecasting (e.g., Temporal Fusion Transformer), easy-to-use API for model training and evaluation. |
| **TensorBoard for PyTorch** | Visualization tool for monitoring PyTorch model training. | TensorBoard integration for visualizing loss curves, metrics, and model graphs during training. |

## 🛣️Roadmap

![PyTorch](https://github.com/mohd-faizy/PyTorch-Essentials/blob/5acd20cf064df9b145ecddf41777001b68d58998/_img/PyTorch-M2.png)

## 📒Colab-Notebook-1

| # | Notebook | Link |
|---|---------|------|
| 01 | **Pytorch Basics P1** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_w9niMqKK9ZEH4MNicD5RNyyWMBZ0o-_?usp=sharing)|
| 02 | **Pytorch Basics P2** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iKrt_8YOMR06oD42J1lp_-XEdrMxv0Dg?usp=sharing)|
| 03 | **Linear Regression** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1v4ZyU-gmVbPD2o7-sbnUYhc06qLzUj-g/view?usp=sharing)|
| 04 | **Binary Classification P1 - [`sklearn` `make_moons` dataset]** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/136APJ-1-yHLYErbL4h0pxV20d8skWuAs?usp=sharing)|
| 05 | **Binary-Classification P2** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/180uA8tOTRn3GgYleUnEk6QD3dneDFS99?usp=sharing)|
| 06 | **Multi-Class Classification P1 - [`sklearn` `make_Blob` dataset]** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_aRQd9oJk5bOsPWyJ80UM1ctxiRFyOaP?usp=sharing)|
| 07 | **Multi-Class-Classification P2** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x_Rfv6ChnFpZyEecnlIEUNT-W6AqR6cB?usp=sharing)|
| 08 | **Computer Vision - Artificial Neural Network (ANN)** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16jEyphHuDEWrhbbRkQe87F6NY48C_vKm?usp=sharing)|
| 09 | **Computer Vision - LeNet5** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d5oQCy2EtElmkgix6ZW6IckRUZieMCmG?usp=sharing)|
| 10 | **Computer Vision - Convolutional Neural Network (CNN)** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1OxHpLJQP7cKAatn76OdNJzGwWfVRA4bM/view?usp=sharing)|
| 11 | **Custom Datasets P1** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1fE02hTsmvQhH768w4e7PGHGWXgIYvNnS/view?usp=sharing)|
| 11 | **Costom_DataSets P2** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LpT0c7fzfzRWA3fEB9BWgEi3JKFqRCzW?usp=sharing)|
| 11 | **PyTorch Going Modular** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SuOM32KgIgQbkWczPapXPY6BQYU6MM9y?usp=sharing)|
| 12 | **Transfer Learning** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)|


## 📒Colab-Notebook-2

| # | Notebook | Link |
|---|---------|------|
| 01 | **Tensor in PyTorch** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1_wy-hcKfMT7v_kpXxJd2VQibbY8qJbFi/view?usp=sharing)|
| 02 | **PyTorch Autograd** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1eHEdnT-gNh5ImCc0giXf32PRI7_i07hX/view?usp=sharing)|
| 03 | **PyTorch Training Pipeline** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Thw9YVUJPKRrzWPWdwGVQJrTbfnE3xMn/view?usp=sharing)|
| 04 | **PyTorch `nn` Module** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Puh7UAb4n9y9yz1puTi7iU2HZFtitXk_/view?usp=sharing)|
| 05 | **Dataset &dataloader** |  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1CkOss-YlvBRnjmRVE45sAErQhl0Ne9HK/view?usp=sharing)|


## 🔦Explore

This repository covers a wide range of topics, including:

- Fundamentals of PyTorch: Learn about **tensors**, **operations**, **autograd**, and **optimization** techniques.
- **GPU Usage Optimization**: Explore strategies to efficiently utilize GPUs for accelerated deep learning workflows.
- Delve into advanced concepts like:
  - **👁️Computer Vision:**  Dive into image classification, object detection, image segmentation, and transfer learning using PyTorch.
  - **🔊Natural Language Processing:** Discover how PyTorch powers state-of-the-art NLP models for tasks like sentiment analysis, language translation, and text generation.
  - **🖼️Generative Models:** Explore how to create entirely new data, like generating realistic images or writing creative text.
  - **🛠️Reinforcement Learning:** Train models to learn optimal strategies through interaction with an environment.

- **Custom Datasets and Data Loading:** Master the art of creating custom datasets and efficient data loading pipelines in PyTorch.
- **Modular Workflows:** Build modular and scalable deep learning pipelines for seamless experimentation and model deployment.
- **Experiment Tracking:** Learn best practices for experiment tracking, model evaluation, and hyperparameter tuning.
- **Replicating Research Papers:** Replicate cutting-edge research papers and implement state-of-the-art deep learning models.
- **Model Deployment:** Explore techniques for deploying PyTorch models in production environments, including cloud deployments and edge devices.

- **Bonus:** Dive into the exciting world of **PyTorch Lightning**, a framework that streamlines the machine learning development process.

## 💧PyTorch code

| Category              | Import Code Example                           | Description                                                                                   | See                                  |
|-----------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------|
| **Imports**           | `import torch`                               | Root package                                                                                  |                                      |
|                       | `from torch.utils.data import Dataset, DataLoader` | Dataset representation and loading                                                            |                                      |
|                       | `import torchvision`                         | Computer vision tools and datasets                                                            | torchvision                          |
|                       | `from torchvision import datasets, models, transforms` | Vision datasets, architectures & transforms                                             | torchvision                          |
|                       | `import torch.nn as nn`                      | Neural networks                                                                               | nn                                   |
|                       | `import torch.nn.functional as F`            | Layers, activations, and more                                                                 | functional                           |
|                       | `import torch.optim as optim`                | Optimizers (e.g., gradient descent, ADAM, etc.)                                               | optim                                |
|                       | `from torch.autograd import Variable`        | For variable management in autograd                                                           | autograd                             |
| **Neural Network API**| `from torch import Tensor`                   | Tensor node in the computation graph                                                          |                                      |
|                       | `import torch.autograd as autograd`          | Computation graph                                                                             | autograd                             |
|                       | `from torch.nn import Module`                | Base class for all neural network modules                                                     | nn                                   |
|                       | `from torch.nn import functional as F`       | Functional interface for neural networks                                                      | functional                           |
| **TorchScript and JIT**| `from torch.jit import script, trace`        | Hybrid frontend decorator and tracing JIT                                                     | TorchScript                          |
|                       | `torch.jit.trace(model, input)`              | Traces computational steps of data input through the model                                    | TorchScript                          |
|                       | `@script`                                    | Decorator indicating data-dependent control flow                                               | TorchScript                          |
| **ONNX**              | `import torch.onnx`                          | ONNX export interface                                                                         | onnx                                 |
|                       | `torch.onnx.export(model, dummy_input, "model.onnx")` | Exports a model to ONNX format using trained model, dummy data, and file name                | onnx                                 |
| **Data Handling**     | `x = torch.randn(*size)`                     | Tensor with independent N(0,1) entries                                                        | tensor                               |
|                       | `x = torch.ones(*size)`                      | Tensor with all 1's                                                                           | tensor                               |
|                       | `x = torch.zeros(*size)`                     | Tensor with all 0's                                                                           | tensor                               |
|                       | `x = torch.tensor(L)`                        | Create tensor from [nested] list or ndarray L                                                 | tensor                               |
|                       | `y = x.clone()`                              | Clone of x                                                                                    | tensor                               |
|                       | `with torch.no_grad():`                      | Code wrap that stops autograd from tracking tensor history                                    | tensor                               |
|                       | `x.requires_grad_(True)`                     | In-place operation, when set to True, tracks computation history for future derivative calculations | tensor                          |
| **Dimensionality**    | `x.size()`                                   | Returns tuple-like object of dimensions                                                       | tensor                               |
|                       | `x = torch.cat(tensor_seq, dim=0)`           | Concatenates tensors along dim                                                                | tensor                               |
|                       | `y = x.view(a, b, ...)`                      | Reshapes x into size (a, b, ...)                                                              | tensor                               |
|                       | `y = x.view(-1, a)`                          | Reshapes x into size (b, a) for some b                                                        | tensor                               |
|                       | `y = x.transpose(a, b)`                      | Swaps dimensions a and b                                                                      | tensor                               |
|                       | `y = x.permute(*dims)`                       | Permutes dimensions                                                                           | tensor                               |
|                       | `y = x.unsqueeze(dim)`                       | Tensor with added axis                                                                        | tensor                               |
|                       | `y = x.squeeze()`                            | Removes all dimensions of size 1                                                              | tensor                               |
|                       | `y = x.squeeze(dim=1)`                       | Removes specified dimension of size 1                                                         | tensor                               |
| **Algebra**           | `ret = A.mm(B)`                              | Matrix multiplication                                                                         | math operations                      |
|                       | `ret = A.mv(x)`                              | Matrix-vector multiplication                                                                  | math operations                      |
|                       | `x = x.t()`                                  | Matrix transpose                                                                              | math operations                      |
| **GPU Usage**         | `torch.cuda.is_available()`                  | Check for CUDA availability                                                                   | cuda                                 |
|                       | `x = x.cuda()`                               | Move x's data from CPU to GPU and return new object                                           | cuda                                 |
|                       | `x = x.cpu()`                                | Move x's data from GPU to CPU and return new object                                           | cuda                                 |
|                       | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` | Device agnostic code                                                                         | cuda                                 |
|                       | `model.to(device)`                           | Recursively convert parameters and buffers to device-specific tensors                         | cuda                                 |
|                       | `x = x.to(device)`                           | Copy tensors to a device (GPU, CPU)                                                           | cuda                                 |
| **Deep Learning**     | `nn.Linear(m, n)`                            | Fully connected layer from m to n units                                                       | nn                                   |
|                       | `nn.Conv2d(m, n, s)`                         | 2-dimensional conv layer from m to n channels with kernel size s                              | nn                                   |
|                       | `nn.MaxPool2d(s)`                            | 2-dimensional max pooling layer                                                               | nn                                   |
|                       | `nn.BatchNorm2d(num_features)`               | Batch normalization layer                                                                     | nn                                   |
|                       | `nn.RNN(input_size, hidden_size)`            | Recurrent Neural Network layer                                                                | nn                                   |
|                       | `nn.LSTM(input_size, hidden_size)`           | Long Short-Term Memory layer                                                                  | nn                                   |
|                       | `nn.GRU(input_size, hidden_size)`            | Gated Recurrent Unit layer                                                                    | nn                                   |
|                       | `nn.Dropout(p=0.5)`                          | Dropout layer                                                                                 | nn                                   |
|                       | `nn.Embedding(num_embeddings, embedding_dim)`| Mapping from indices to embedding vectors                                                     | nn                                   |
| **Loss Functions**    | `nn.CrossEntropyLoss()`                      | Cross-entropy loss                                                                            | loss functions                       |
|                       | `nn.MSELoss()`                               | Mean Squared Error loss                                                                       | loss functions                       |
|                       | `nn.NLLLoss()`                               | Negative Log-Likelihood loss                                                                  | loss functions                       |
| **Activation Functions** | `nn.ReLU()`                              | Rectified Linear Unit activation function                                                     | activation functions                 |
|                       | `nn.Sigmoid()`                               | Sigmoid activation function                                                                   | activation functions                 |
|                       | `nn.Tanh()`                                  | Tanh activation function                                                                      | activation functions                 |
| **Optimizers**        | `optimizer = optim.SGD(model.parameters(), lr=0.01)` | Stochastic Gradient Descent optimizer                                                         | optimizers                           |
|                       | `optimizer = optim.Adam(model.parameters(), lr=0.001)`| ADAM optimizer                                                                                | optimizers                           |
|                       | `optimizer.step()`                           | Update weights                                                                                | optimizers                           |
| **Learning Rate Scheduling** | `scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)` | Create learning rate scheduler                                                               | learning rate scheduler  |
|                       | `scheduler.step()`                           | Adjust learning rate                                                                          | learning rate scheduler              |

## ⚡PyTorch APIs

![PyTorch-APIs](https://github.com/mohd-faizy/PyTorch-Essentials/blob/main/_img/PyTorch-APIs.png)

## ⭐Who uses PyTorch

| **Company**          | **Products/Services Using PyTorch**                           | **Technical Description of Usage**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-----------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Meta Platforms (Facebook)** | -> **Facebook App**<br> -> **Instagram**<br> -> **Meta AI Research Projects** | ***Meta developed PyTorch to meet its need for a flexible and developer-friendly deep learning framework. Specific usages include:***<br>- **Computer Vision**: Real-time image and video analysis for facial recognition, content moderation, and augmented reality (AR) on platforms like Facebook and Instagram.<br>- **Natural Language Processing (NLP)**: Text understanding for translations, chatbots, and content ranking.<br>- **AI Research**: Meta’s FAIR (Facebook AI Research) group uses PyTorch to prototype advanced algorithms, including reinforcement learning, GANs (Generative Adversarial Networks), and transformers.<br>- **TorchServe**: For serving machine learning models in production. |
| **Microsoft**         | -> **Azure Machine Learning**<br> -> **Bing Search**<br> -> **Office 365 Intelligent Features** | ***Microsoft integrates PyTorch as a core framework in several domains:***<br>- **Azure AI Services**: Azure offers PyTorch as a primary framework for training and deploying machine learning models. This includes AutoML, custom vision APIs, and reinforcement learning tools.<br>- **Bing Search**: PyTorch is utilized in ranking algorithms for search relevance, improving user query understanding through transformer-based models like BERT.<br>- **Office 365 AI Features**: AI-driven enhancements such as Excel’s predictive analytics, Outlook’s Smart Replies, and grammar suggestions are powered by PyTorch-backed NLP models.<br>- **ONNX Runtime**: Microsoft uses ONNX (Open Neural Network Exchange) to optimize PyTorch models for cross-platform deployment. |
| **Tesla**             | -> **Autopilot System**<br> -> **Full Self-Driving (FSD) Capability** | ***Tesla employs PyTorch to train deep learning models that are integral to autonomous vehicle technology:***<br>- **Computer Vision**: PyTorch is used to develop convolutional neural networks (CNNs) for tasks like lane detection, traffic sign recognition, and obstacle avoidance.<br>- **Sensor Fusion**: PyTorch helps combine data from cameras, LIDAR, radar, and ultrasonic sensors to create a cohesive 3D representation of the vehicle's surroundings.<br>- **End-to-End Neural Networks**: Tesla relies on PyTorch to build perception and decision-making networks capable of navigating complex environments.<br>- **Simulation Training**: PyTorch powers Tesla’s simulation environments, where millions of miles are virtually driven to improve safety and performance. |
| **OpenAI**            | -> **GPT Models**<br> -> **DALL·E**<br> -> **ChatGPT**              | ***OpenAI uses PyTorch as the foundation for its advanced AI research and development:***<br>- **Large-Scale Language Models**: GPT (Generative Pre-trained Transformer) models, including GPT-4, are trained on PyTorch for tasks such as language generation, summarization, and code understanding.<br>- **DALL·E**: PyTorch supports multimodal models that generate images from textual descriptions, leveraging architectures like CLIP and diffusion models.<br>- **ChatGPT**: Combines PyTorch-trained transformer architectures with reinforcement learning from human feedback (RLHF) for conversational AI.<br>- **Distributed Training**: OpenAI extensively uses PyTorch’s `torch.distributed` package for parallel training across GPUs and nodes, enabling the efficient training of models with billions of parameters. |
| **Uber**              | -> **Uber Ride-Hailing Platform**<br> -> **Uber Eats Recommendations**<br> -> **Pyro (Probabilistic Programming)** | ***Uber utilizes PyTorch across multiple domains:***<br>- **Demand Forecasting and Route Optimization**: PyTorch aids in building deep learning models for predicting ride demand and optimizing driver routes using reinforcement learning.<br>- **Recommendations**: Uber Eats employs PyTorch for personalized restaurant and dish recommendations using collaborative filtering and graph neural networks (GNNs).<br>- **Pyro**: Uber developed Pyro, a probabilistic programming library built on PyTorch, for Bayesian inference, causal modeling, and probabilistic machine learning. This is particularly useful for decision-making under uncertainty.<br>- **Autonomous Mobility**: Uber ATG (Advanced Technologies Group) used PyTorch for perception and control models in its self-driving car research. |


## 🍰 Contributing

Contributions are welcome!

## 🙇 Acknowledgements

- A heartfelt thank you to the following resources that inspired this Repository:
  - [PyTorch.org](https://pytorch.org)
  - [mrdbourke](https://github.com/mrdbourke/pytorch-deep-learning)
  - [Udacity](https://learn.udacity.com/my-programs?tab=Currently%2520Learning)
  - [Lazy Programmer](https://www.udemy.com/course/pytorch-deep-learning/)

## ⚖ ➤ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## ❤️ Support

If you find this repository helpful, show your support by starring it! For questions or feedback, reach out on [Twitter(`X`)](https://twitter.com/F4izy).

## 🔗Connect with me:

➤ If you have questions or feedback, feel free to reach out!!!

[<img align="left" src="https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/twitter_circle-512.png" width="32px"/>][twitter]
[<img align="left" src="https://cdn-icons-png.flaticon.com/512/145/145807.png" width="32px"/>][linkedin]
[<img align="left" src="https://cdn-icons-png.flaticon.com/512/2626/2626299.png" width="32px"/>][Portfolio]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/mohd-faizy/
[Portfolio]: https://ai.stackexchange.com/users/36737/faizy?tab=profile

---

<img src="https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true" width=380px height=200px />
