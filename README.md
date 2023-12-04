# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:39:21 2023

@author: jland
"""

# Tumor detection

## Overview

This project is a binary image classification task using a Convolutional Neural Network (CNN) implemented in Keras. The goal is to detect the presence or absence of tumors in medical images.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data Visualization](#data-visualization)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Testing](#testing)
- [Sample Image Visualization](#sample-image-visualization)
- [Image Resizing](#image-resizing)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

In this project, we leverage a CNN architecture to perform binary image classification on medical images. The model is trained to distinguish between images with tumors ('yes') and those without tumors ('no').

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/yourusername/your-repo.git

2. Install the required dependencies.
   ```bash
      pip install -r requirements.txt

3. Explore the codebase and datasets.

## Usage

To train the model, use the following command:
   ```bash
      git python main_script.py

This script will preprocess the data, train the model, and generate visualizations.

## Data Visualization
Explore the distribution of images before and after splitting the dataset by using the.

visualize_image_distribution and visualize_data functions, respectively.

## Model Training

The model consists of convolutional layers, max-pooling layers, and dense layers. It is trained using the Adam optimizer and binary cross-entropy loss.

## Model Evaluation

Evaluate the model on the validation set and visualize the training and validation accuracy/loss using the generated plots. A confusion matrix is also displayed to assess the model's performance.

## Testing

Test the trained model on the provided test set and view the test loss and accuracy.

## Sample Image Visualization

Visualize a grid of sample images from the training set using the visualize_img function.

```bash
      python show_img.py

## Image Resizing

Resize images in the dataset using the resize_images function.

```bash
   python normalize.py

## Dependencies

Python (>=3.6)
Keras (>=2.0)
Matplotlib (>=3.0)
Pillow (>=7.0)
Seaborn (>=0.9)

