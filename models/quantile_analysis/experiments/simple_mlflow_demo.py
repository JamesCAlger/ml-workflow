#!/usr/bin/env python3
"""
Simple MLflow Demonstration

This script demonstrates basic MLflow tracking functionality
without complex hyperparameter optimization.
"""

import sys
sys.path.append('src')

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

def simple_mlflow_demo():
    """Demonstrate basic MLflow tracking"""
    
    print("🚀 Simple MLflow Tracking Demonstration")
    print("=" * 50)
    
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("simple_demo")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * np.random.randn(n_samples)
    
    # Convert to DataFrame for better visualization
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Created synthetic dataset: {df.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name="simple_demo_run") as run:
        print(f"\n📊 Started MLflow run: {run.info.run_id}")
        
        # Log parameters
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        
        # Train model
        print("🔧 Training Random Forest model...")
        start_time = time.time()
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Log feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Create and log artifacts
        # Data summary
        summary_text = f"""
Experiment Summary
==================
Dataset shape: {df.shape}
Train samples: {len(X_train)}
Test samples: {len(X_test)}
Features: {feature_names}

Model Performance:
- Train MSE: {train_mse:.4f}
- Test MSE: {test_mse:.4f}
- Train MAE: {train_mae:.4f}  
- Test MAE: {test_mae:.4f}
- Training time: {training_time:.2f} seconds

Feature Importance:
{chr(10).join([f'- {k}: {v:.4f}' for k, v in feature_importance.items()])}
        """
        
        with open("experiment_summary.txt", "w") as f:
            f.write(summary_text)
        mlflow.log_artifact("experiment_summary.txt")
        
        # Log dataset info
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.head(10).to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "data_samples")
        
        print("📈 Results:")
        print(f"  Train MSE: {train_mse:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")
        
        print(f"\n✅ MLflow run completed: {run.info.run_id}")
        print(f"📊 View results at: http://localhost:5000")

def multiple_experiments_demo():
    """Demonstrate tracking multiple experiments"""
    
    print("\n🔄 Multiple Experiments Demonstration")
    print("=" * 50)
    
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("multiple_models_comparison")
    
    # Create data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * np.random.randn(n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test different hyperparameters
    model_configs = [
        {"n_estimators": 50, "max_depth": 5, "name": "small_model"},
        {"n_estimators": 100, "max_depth": 10, "name": "medium_model"},
        {"n_estimators": 200, "max_depth": 15, "name": "large_model"},
    ]
    
    results = []
    
    for config in model_configs:
        with mlflow.start_run(run_name=config["name"]) as run:
            print(f"🔧 Training {config['name']}...")
            
            # Log parameters
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", config["max_depth"])
            mlflow.log_param("model_type", "RandomForest")
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            test_mse = mean_squared_error(y_test, model.predict(X_test))
            test_mae = mean_absolute_error(y_test, model.predict(X_test))
            
            # Log metrics
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_mae", test_mae)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            results.append({
                "name": config["name"],
                "run_id": run.info.run_id,
                "test_mse": test_mse,
                "test_mae": test_mae
            })
            
            print(f"  {config['name']}: MSE={test_mse:.4f}, MAE={test_mae:.4f}")
    
    # Find best model
    best_model = min(results, key=lambda x: x["test_mse"])
    print(f"\n🏆 Best model: {best_model['name']} (MSE: {best_model['test_mse']:.4f})")
    print(f"   Run ID: {best_model['run_id']}")

def main():
    """Main function"""
    
    # Check if MLflow UI is accessible
    try:
        import requests
        response = requests.get("http://localhost:5000", timeout=2)
        ui_status = "✅ Running"
    except:
        ui_status = "❌ Not running (run 'mlflow ui --port 5000')"
    
    print("🌐 MLflow UI Status:", ui_status)
    print("📍 MLflow UI URL: http://localhost:5000")
    print()
    
    # Run demonstrations
    simple_mlflow_demo()
    multiple_experiments_demo()
    
    print("\n🎯 Summary:")
    print("- ✅ Basic MLflow tracking demonstrated")
    print("- ✅ Multiple experiments logged and compared") 
    print("- ✅ Parameters, metrics, and artifacts tracked")
    print("- ✅ Models saved and versioned")
    print(f"- 🌐 View all results at: http://localhost:5000")
    
    print("\n💡 Next steps:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. Explore the 'simple_demo' and 'multiple_models_comparison' experiments")
    print("3. Compare runs, view metrics, and download models")

if __name__ == "__main__":
    main() 