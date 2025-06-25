import os
import yaml
import json
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results, output_path, filename="results.json"):
    """Save results to JSON file"""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    if isinstance(results, dict):
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):
                serializable_results[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
    else:
        serializable_results = results
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

def save_models(models, output_path):
    """Save trained models to pickle files"""
    models_dir = os.path.join(output_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    for quantile, model in models.items():
        model_filename = f"model_q{int(quantile*100)}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

def load_models(models_path):
    """Load trained models from pickle files"""
    models = {}
    for filename in os.listdir(models_path):
        if filename.startswith("model_q") and filename.endswith(".pkl"):
            # Extract quantile from filename
            quantile_str = filename.replace("model_q", "").replace(".pkl", "")
            quantile = float(quantile_str) / 100
            
            model_path = os.path.join(models_path, filename)
            with open(model_path, 'rb') as f:
                models[quantile] = pickle.load(f)
    
    return models

def create_experiment_dir(base_path, experiment_name):
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_path, f"{experiment_name}_{timestamp}")
    
    # Create subdirectories
    subdirs = ["models", "predictions", "plots"]
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir

def save_config_copy(config, output_path, filename="config_used.yaml"):
    """Save copy of configuration used for experiment"""
    config_path = os.path.join(output_path, filename)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def save_predictions(predictions, output_path, filename="predictions.csv"):
    """Save predictions to CSV file"""
    predictions_dir = os.path.join(output_path, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Convert predictions dict to DataFrame
    if isinstance(predictions, dict):
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(predictions_dir, filename), index=False)
    else:
        predictions.to_csv(os.path.join(predictions_dir, filename), index=False)

def get_absolute_path(relative_path):
    """Convert relative path to absolute path from project root"""
    # Get the actual location of this utils.py file and resolve symlinks
    utils_file = Path(__file__).resolve()
    # From models/quantile_analysis/src/utils.py, go up to project root (4 levels)
    project_root = utils_file.parent.parent.parent.parent
    absolute_path = project_root / relative_path
    return absolute_path

def validate_config(config, required_keys):
    """Validate that config contains required keys"""
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

def pinball_loss(y_true, y_pred, quantile):
    """Calculate pinball loss for a given quantile"""
    import numpy as np
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))

def empirical_coverage(y_true, lower_bound, upper_bound):
    """Calculate empirical coverage of prediction intervals"""
    import numpy as np
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return coverage

def prediction_interval_score(y_true, lower_bound, upper_bound, alpha):
    """Calculate prediction interval score (PIS)"""
    import numpy as np
    # Width of interval
    width = upper_bound - lower_bound
    
    # Penalties for being outside the interval
    lower_penalty = (2/alpha) * (lower_bound - y_true) * (y_true < lower_bound)
    upper_penalty = (2/alpha) * (y_true - upper_bound) * (y_true > upper_bound)
    
    pis = width + lower_penalty + upper_penalty
    return np.mean(pis) 