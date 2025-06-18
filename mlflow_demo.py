#!/usr/bin/env python3
"""
MLflow Demo Script - Wine Quality Prediction with Large File Tracking
=====================================================================

This script demonstrates how to use MLflow for:
1. Tracking large model files
2. Logging artifacts and datasets
3. Model registry management
4. Experiment organization

Author: Jagadesh Chilla
Project: ANN-with-ML-flow
"""

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
from mlflow.models import infer_signature
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def setup_mlflow():
    """Setup MLflow tracking and experiment"""
    # Set experiment name
    experiment_name = "wine-quality-production"
    mlflow.set_experiment(experiment_name)
    
    # Set tracking URI (use local by default)
    mlflow.set_tracking_uri("file:./mlruns")
    
    print(f"✅ MLflow experiment '{experiment_name}' is ready!")
    print(f"📊 Tracking URI: {mlflow.get_tracking_uri()}")

def load_and_preprocess_data():
    """Load and preprocess the wine quality dataset"""
    print("📥 Loading wine quality dataset...")
    
    # Load data
    url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
    data = pd.read_csv(url, sep=";")
    
    # Split features and target
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset shape: {data.shape}")
    print(f"🎯 Training samples: {X_train.shape[0]}")
    print(f"🧪 Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, data

def create_and_train_model(X_train, y_train, X_test, y_test):
    """Create and train the neural network model"""
    print("🧠 Creating and training neural network...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate model
    test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"📈 Model trained successfully!")
    print(f"🎯 Test MAE: {test_mae:.4f}")
    print(f"🎯 Test MSE: {test_mse:.4f}")
    
    return model, scaler, history, test_mae, test_mse, X_train_scaled, y_train

def log_with_mlflow(model, scaler, history, test_mae, test_mse, X_train, y_train, data):
    """Log model, artifacts, and metrics with MLflow"""
    print("📝 Logging to MLflow...")
    
    with mlflow.start_run(run_name="wine-quality-production-model") as run:
        # Log parameters
        mlflow.log_param("model_type", "Neural Network")
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("architecture", "128-64-32-1")
        mlflow.log_param("dataset_size", len(data))
        
        # Log metrics
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", np.sqrt(test_mse))
        
        # Log training history
        for epoch, (loss, mae, val_loss, val_mae) in enumerate(zip(
            history.history['loss'],
            history.history['mae'],
            history.history['val_loss'],
            history.history['val_mae']
        )):
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_mae", mae, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_mae", val_mae, step=epoch)
        
        # Create signature for model
        signature = infer_signature(X_train, y_train)
        
        # Log the model (this handles large model files automatically)
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="wine-quality-nn"
        )
        
        # Save and log the scaler as an artifact
        scaler_path = "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, "preprocessing")
        
        # Save and log dataset info
        data_info = {
            "dataset_url": "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
            "features": list(data.columns[:-1]),
            "target": "quality",
            "dataset_shape": data.shape,
            "feature_count": len(data.columns) - 1
        }
        
        # Save dataset info as JSON
        import json
        with open("dataset_info.json", "w") as f:
            json.dump(data_info, f, indent=2, default=str)
        mlflow.log_artifact("dataset_info.json", "data")
        
        # Log model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        with open("model_summary.txt", "w") as f:
            f.write("\n".join(model_summary))
        mlflow.log_artifact("model_summary.txt", "model_info")
        
        # Clean up temporary files
        os.remove(scaler_path)
        os.remove("dataset_info.json")
        os.remove("model_summary.txt")
        
        print(f"✅ Successfully logged to MLflow!")
        print(f"🔗 Run ID: {run.info.run_id}")
        print(f"📊 Experiment ID: {run.info.experiment_id}")
        
        return run.info.run_id

def demonstrate_model_loading(run_id):
    """Demonstrate how to load the logged model"""
    print("🔄 Demonstrating model loading...")
    
    # Load model using run ID
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Create sample data for prediction
    sample_data = np.array([[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]])
    
    # Make prediction
    prediction = loaded_model.predict(sample_data)
    
    print(f"🎯 Sample prediction: {prediction[0]:.2f}")
    print(f"✅ Model loading successful!")

def main():
    """Main execution function"""
    print("🍷 Wine Quality Prediction with MLflow - Large File Demo")
    print("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, data = load_and_preprocess_data()
    
    # Create and train model
    model, scaler, history, test_mae, test_mse, X_train_scaled, y_train = create_and_train_model(
        X_train, y_train, X_test, y_test
    )
    
    # Log with MLflow
    run_id = log_with_mlflow(model, scaler, history, test_mae, test_mse, X_train_scaled, y_train, data)
    
    # Demonstrate model loading
    demonstrate_model_loading(run_id)
    
    print("\n🎉 MLflow demo completed successfully!")
    print("💡 Tips for large files:")
    print("   • MLflow automatically handles large model files")
    print("   • Use mlflow.log_artifact() for large datasets")
    print("   • Consider using remote storage (S3, Azure, GCS) for production")
    print("   • Use model registry for version management")
    print("\n📊 To view results, run: mlflow ui")

if __name__ == "__main__":
    main() 