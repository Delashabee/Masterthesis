# 3D Scanning and Modelling of Unknown Objects with a Two-Arm Robotic System

This repository supports my research on 3D scanning and modeling of unknown objects using a dual-arm robotic system. The code and tools provided here are designed to facilitate data generation, training, and evaluation processes essential for building robust 3D object models.

## Datagenerator

The `Datagenerator` directory contains programs that generate training data. This module can operate in multiple modes to create different types of data outputs:

- **Voxel Data Generation**
- **Point Cloud Data Generation**
- **Coverage Calculation**: Computes coverage metrics to evaluate data completeness.

### Requirements
To use the programs in this repository, ensure that the following libraries are installed:

- OpenCV 4.4.0
- PCL (Point Cloud Library) 1.9.1
- Eigen 3.3.9
- OctoMap 1.9.6
- Gurobi 9.1.1 (Note: Gurobi is only free for academic use)

The code can be compiled using **Visual Studio 2019** with support for **C++ 14** and is intended to run on **Windows 10**. For use on other systems, please verify compatibility, particularly with file reading/writing functions and multithreading features, as adjustments may be needed.

## GVPN
This directory contains my neural network implementations. The model.py file stores several neural network models, and running this file directly allows you to test the output results of the models. The dataset.py script is designed to preprocess the training data. You can train the neural networks by running train.py.
