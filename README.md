# 🍷 Wine Quality Prediction with MLflow & Neural Networks

[![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.19.0-orange.svg)](https://tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-v3.1.0-green.svg)](https://mlflow.org/)
[![Keras](https://img.shields.io/badge/Keras-v3.10.0-red.svg)](https://keras.io/)
[![Hyperopt](https://img.shields.io/badge/Hyperopt-v0.2.7-purple.svg)](http://hyperopt.github.io/hyperopt/)

A comprehensive machine learning project that predicts wine quality using Artificial Neural Networks (ANN) with MLflow for experiment tracking, model management, and deployment.

## 📊 Project Overview

This project demonstrates a complete ML workflow for wine quality prediction using:
- **Deep Learning**: Artificial Neural Networks with Keras/TensorFlow
- **Hyperparameter Optimization**: Bayesian optimization with Hyperopt
- **Experiment Tracking**: MLflow for comprehensive experiment management
- **Model Registry**: MLflow model versioning and deployment
- **Data**: Wine Quality Dataset from UCI ML Repository

![MLflow Architecture](Screenshot%202025-06-18%20114110-1.png)

## 🚀 Features

- ✅ **Automated Hyperparameter Tuning** using Bayesian Optimization
- ✅ **Experiment Tracking** with MLflow UI
- ✅ **Model Versioning** and Registry Management
- ✅ **REST API Deployment** ready models
- ✅ **Reproducible Results** with seed management
- ✅ **Data Preprocessing** with normalization layers
- ✅ **Performance Metrics** tracking (RMSE, Loss)
- ✅ **Model Comparison** across different hyperparameter configurations

## 📈 Model Performance

The neural network achieves optimal performance with:
- **Best RMSE**: ~1.22 on validation set
- **Architecture**: Input → Normalization → Dense(64, ReLU) → Dense(1)
- **Optimizer**: SGD with optimized learning rate and momentum
- **Training**: 3 epochs with early convergence

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **ML Framework** | TensorFlow/Keras | 2.19.0 |
| **Experiment Tracking** | MLflow | 3.1.0 |
| **Hyperparameter Optimization** | Hyperopt | 0.2.7 |
| **Data Processing** | Pandas, NumPy | Latest |
| **Model Evaluation** | Scikit-learn | 1.7.0 |
| **Environment** | Python | 3.10+ |

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/jagadeshchilla/ANN-with-ML-flow.git
   cd ANN-with-ML-flow
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook starter.ipynb
   ```

## 🎯 Quick Start

### 1. Data Loading & Preprocessing
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load wine quality dataset
data = pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv", sep=";")

# Split data
train, test = train_test_split(data, test_size=0.25, random_state=42)
```

### 2. MLflow Experiment Setup
```python
import mlflow

# Set experiment
mlflow.set_experiment("wine-quality")

# Start tracking
with mlflow.start_run():
    # Your ML code here
    pass
```

### 3. Hyperparameter Optimization
```python
from hyperopt import fmin, tpe, hp, Trials

# Define search space
space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0)
}

# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=4)
```

## 📊 MLflow UI Dashboard

Launch the MLflow UI to visualize experiments:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to access:
- **Experiment Comparison**: Compare runs side-by-side
- **Metrics Visualization**: Track RMSE, loss over time
- **Parameter Analysis**: Understand hyperparameter impact
- **Model Registry**: Manage model versions

## 🏗️ Project Structure

```
ANN-with-ML-flow/
├── starter.ipynb              # Main notebook with complete workflow
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── Screenshot*.png           # MLflow UI screenshots
├── venv/                     # Virtual environment
└── mlruns/                   # MLflow experiment artifacts
    ├── experiments/
    ├── models/
    └── artifacts/
```

## 🔬 Experiment Results

### Hyperparameter Optimization Results

| Run | Learning Rate | Momentum | RMSE | Status |
|-----|---------------|----------|------|--------|
| 1   | 0.0028       | 0.98     | **1.22** | ✅ Best |
| 2   | 0.0156       | 0.45     | 1.38 | ✅ |
| 3   | 0.0089       | 0.72     | 4.05 | ✅ |
| 4   | 0.0234       | 0.91     | 4.74 | ✅ |

### Model Architecture
```
Model: Sequential
├── Input Layer (11 features)
├── Normalization Layer
├── Dense Layer (64 units, ReLU)
└── Output Layer (1 unit, Linear)

Total params: 833
Trainable params: 833
```

## 🚀 Model Deployment

### REST API Deployment
```bash
# Serve model as REST API
mlflow models serve -m "models:/wine-quality/1" -p 1234

# Make predictions
curl -X POST -H "Content-Type:application/json" \
  --data '{"inputs": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]}' \
  http://localhost:1234/invocations
```

### Docker Deployment
```bash
# Build Docker image
mlflow models build-docker -m "models:/wine-quality/1" -n wine-quality-model

# Run container
docker run -p 8080:8080 wine-quality-model
```

## 📈 Performance Metrics

- **Training Loss**: Converges within 3 epochs
- **Validation RMSE**: 1.22 (Best model)
- **Model Size**: ~3.3KB (highly efficient)
- **Inference Time**: <1ms per prediction
- **Data Coverage**: 4,898 wine samples

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Acknowledgments

- **Dataset**: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UCI ML Repository
- **MLflow**: Open-source ML lifecycle management platform
- **Hyperopt**: Bayesian optimization library
- **TensorFlow**: End-to-end ML platform

## 📧 Contact

**Jagadesh Chilla** - [GitHub Profile](https://github.com/jagadeshchilla)

Project Link: [https://github.com/jagadeshchilla/ANN-with-ML-flow](https://github.com/jagadeshchilla/ANN-with-ML-flow)

---

⭐ **Star this repository if you found it helpful!** ⭐ 