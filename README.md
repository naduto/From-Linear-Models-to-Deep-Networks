## From Linear Models to Deep Networks

A comprehensive exploration of machine learning classification techniques, progressing from traditional linear models to modern deep neural networks. This project implements and compares various approaches for handwritten digit recognition using the MNIST dataset.

## 📋 Project Overview

This assignment demonstrates the evolution of classification methods through hands-on implementation and analysis:

- **Linear Classification**: Traditional machine learning approach using logistic regression
- **Neural Networks**: Deep learning implementation with feedforward architecture
- **Hyperparameter Optimization**: Systematic analysis of model configuration impact
- **Model Comparison**: Performance evaluation across different architectures

## 🎯 Learning Objectives

- Understand the fundamentals of linear classification models
- Build and train neural networks from scratch using PyTorch
- Perform hyperparameter tuning and analyze its effects
- Compare model architectures and evaluate trade-offs
- Visualize learning curves and convergence behavior

## 📁 Repository Structure

```
Assignment-2-From-Linear-Models-to-Deep-Networks/
├── Linear Classification Model.ipynb    # Baseline linear model implementation
├── Neural Network.ipynb                 # Feedforward NN architecture & training
├── C1_Hyperparameter_Analysis.ipynb     # Systematic hyperparameter experiments
├── C2_Model_Comparison.ipynb            # Architecture comparison framework
├── data/                                # MNIST dataset (auto-downloaded)
│   └── MNIST/
├── .venv/                               # Python virtual environment
└── README.md                            # Project documentation
```


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (`venv` or `virtualenv`)
- Required packages: PyTorch, NumPy, Matplotlib, scikit-learn

### Installation

1. **Clone the repository**
```shell script
git clone <repository-url>
   cd Assignment-2-From-Linear-Models-to-Deep-Networks
```


2. **Set up virtual environment**
```shell script
python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```


3. **Install dependencies**
```shell script
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```


4. **Launch Jupyter**
```shell script
jupyter notebook
```


## 📓 Notebook Guide

### 1. Linear Classification Model
**Objective**: Establish a baseline using traditional ML techniques

- Implements logistic regression for multi-class classification
- Feature engineering and data preprocessing
- Model training with gradient descent
- Performance evaluation and visualization

**Key Concepts**: Softmax regression, cross-entropy loss, decision boundaries

### 2. Neural Network
**Objective**: Build a feedforward neural network from fundamentals

- Architecture design (input → hidden layers → output)
- Forward and backward propagation implementation
- Xavier initialization for stable training
- Training loop with validation monitoring
- Learning curves and convergence analysis

**Key Concepts**: Activation functions (ReLU), weight initialization, backpropagation

### 3. Hyperparameter Analysis (C1)
**Objective**: Systematically explore hyperparameter impact

Experiments include:
- **Learning rate**: Effect on convergence speed and stability
- **Batch size**: Training dynamics and memory efficiency
- **Network depth**: Layer count vs. performance trade-offs
- **Hidden units**: Width vs. representational capacity
- **Epochs**: Training duration and overfitting

**Deliverables**: Comparative plots, optimal configuration recommendations

### 4. Model Comparison (C2)
**Objective**: Evaluate different architectures

- Performance benchmarking across model variants
- Accuracy vs. computational cost analysis
- Generalization capability assessment
- Model selection guidelines

## 📊 Dataset: MNIST

The Modified National Institute of Standards and Technology (MNIST) database contains:
- **Training set**: 60,000 grayscale images (28×28 pixels)
- **Test set**: 10,000 images
- **Classes**: Handwritten digits (0-9)
- **Split**: 60% train / 20% validation / 20% test (in notebooks)

Dataset is automatically downloaded on first run.

## 🔬 Methodology

### Training Pipeline
1. **Data Preparation**: Load, normalize, and split MNIST data
2. **Model Definition**: Specify architecture and initialization
3. **Training Loop**: Iterate over epochs with mini-batch gradient descent
4. **Validation**: Monitor performance on held-out data
5. **Evaluation**: Test set accuracy and error analysis
6. **Visualization**: Plot loss curves, accuracy metrics, and confusion matrices

### Evaluation Metrics
- **Accuracy**: Classification correctness percentage
- **Loss**: Cross-entropy loss tracking
- **Convergence**: Training/validation gap analysis

## 📈 Key Findings

*(To be completed after experiments)*

Typical results:
- **Linear Model**: ~92% test accuracy (baseline)
- **Neural Network**: ~96-98% test accuracy
- **Optimal Learning Rate**: 0.001-0.01 range
- **Optimal Architecture**: 2-3 hidden layers with 128-256 units

## 🛠️ Troubleshooting

### Common Issues

**1. NotImplementedError: Module missing "forward" function**
- Ensure the neural network class defines a `forward(self, x)` method
- Re-run the cell defining the model class after kernel restarts

**2. CUDA out of memory**
- Reduce batch size in DataLoader
- Decrease model size (fewer layers/units)

**3. Poor convergence**
- Adjust learning rate (try 0.01, 0.001, 0.0001)
- Check data normalization
- Verify weight initialization (Xavier/He initialization)

**4. Overfitting (high train accuracy, low validation)**
- Reduce model complexity
- Implement dropout or regularization
- Increase training data or use data augmentation
