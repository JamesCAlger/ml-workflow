"""
MLflow Client for Modular Experiment Tracking

Provides a clean interface to MLflow functionality while maintaining
modularity and separation of concerns.
"""

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
from mlflow import MlflowClient as MLflowBaseClient
from mlflow.models.signature import infer_signature
import yaml
import os
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MLflowClient:
    """Modular MLflow client for experiment tracking and model management"""
    
    def __init__(self, config_path: str = "config/mlflow_config.yaml"):
        """Initialize MLflow client with configuration"""
        self.config = self._load_config(config_path)
        self._setup_mlflow()
        self._setup_experiment()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MLflow configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load MLflow config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default MLflow configuration"""
        return {
            'mlflow': {
                'tracking_uri': 'sqlite:///mlflow.db',
                'experiment_name': 'quantile_analysis',
                'auto_log': {'lightgbm': True, 'xgboost': True, 'sklearn': True}
            }
        }
    
    def _setup_mlflow(self):
        """Configure MLflow tracking"""
        mlflow_config = self.config.get('mlflow', {})
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Setup auto-logging
        auto_log = mlflow_config.get('auto_log', {})
        if auto_log.get('lightgbm', False):
            mlflow.lightgbm.autolog()
        if auto_log.get('xgboost', False):
            mlflow.xgboost.autolog()
        if auto_log.get('sklearn', False):
            mlflow.sklearn.autolog()
            
        # Initialize MLflow client
        self.client = MLflowBaseClient()
        
        logger.info(f"MLflow configured with tracking URI: {tracking_uri}")
    
    def _setup_experiment(self):
        """Setup or get MLflow experiment"""
        experiment_name = self.config['mlflow'].get('experiment_name', 'quantile_analysis')
        
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
                
            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise
    
    def start_run(self, run_name: str, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        tags = tags or {}
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_experiment_config(self, experiment_config: Dict[str, Any]):
        """Log experiment configuration"""
        # Log individual parameters
        for key, value in experiment_config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))
        
        # Log full config as artifact
        config_path = "experiment_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(experiment_config, f)
        mlflow.log_artifact(config_path)
        os.remove(config_path)
    
    def log_data_quality(self, data: pd.DataFrame, stage: str = "raw"):
        """Log data quality metrics"""
        metrics = {
            f"{stage}_rows": len(data),
            f"{stage}_columns": len(data.columns),
            f"{stage}_missing_values": data.isnull().sum().sum(),
            f"{stage}_missing_percentage": (data.isnull().sum().sum() / data.size) * 100,
            f"{stage}_duplicates": data.duplicated().sum(),
        }
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
    
    def log_transformation_info(self, transformations: Dict[str, Any]):
        """Log transformation metadata"""
        for transform_name, transform_info in transformations.items():
            mlflow.log_param(f"transform_{transform_name}", str(transform_info))
    
    def log_model_performance(self, metrics: Dict[str, float], prefix: str = ""):
        """Log model performance metrics"""
        for metric_name, value in metrics.items():
            full_name = f"{prefix}_{metric_name}" if prefix else metric_name
            mlflow.log_metric(full_name, value)
    
    def log_feature_importance(self, importance_dict: Dict[str, float]):
        """Log feature importance"""
        for feature, importance in importance_dict.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
    
    def log_quantile_predictions(self, predictions: Dict[float, np.ndarray], 
                                 actual: np.ndarray, prefix: str = ""):
        """Log quantile-specific predictions and metrics"""
        for quantile, pred in predictions.items():
            q_str = f"q{int(quantile*100)}"
            
            # Log pinball loss for this quantile
            pinball_loss = self._calculate_pinball_loss(actual, pred, quantile)
            metric_name = f"{prefix}_pinball_loss_{q_str}" if prefix else f"pinball_loss_{q_str}"
            mlflow.log_metric(metric_name, pinball_loss)
    
    def _calculate_pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """Calculate pinball loss for a quantile"""
        residual = y_true - y_pred
        return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
    
    def log_model_artifacts(self, model_dict: Dict[float, Any], model_type: str):
        """Log trained models as artifacts"""
        # Create temporary directory for models
        import tempfile
        import pickle
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_paths = {}
            
            for quantile, model in model_dict.items():
                q_str = f"q{int(quantile*100)}"
                model_filename = f"{model_type}_model_{q_str}.pkl"
                model_path = os.path.join(temp_dir, model_filename)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                model_paths[quantile] = model_path
            
            # Log all model files
            mlflow.log_artifacts(temp_dir, "models")
    
    def log_visualization(self, fig, filename: str):
        """Log matplotlib/plotly figures"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(tmp.name, "visualizations")
            os.unlink(tmp.name)
    
    def register_model(self, model_uri: str, model_name: str, 
                      stage: str = "Staging") -> Any:
        """Register model in MLflow Model Registry"""
        try:
            model_registry = self.config['mlflow'].get('model_registry', {})
            if not model_registry.get('enabled', True):
                logger.info("Model registry disabled in config")
                return None
            
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition to specified stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            
            logger.info(f"Model registered: {model_name} v{model_version.version} -> {stage}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def search_runs(self, filter_string: str = "", max_results: int = 100) -> List[Any]:
        """Search MLflow runs with filters"""
        return self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )
    
    def get_best_run(self, metric_name: str, mode: str = "min") -> Optional[Any]:
        """Get the best run based on a metric"""
        runs = self.search_runs()
        if not runs:
            return None
        
        # Sort runs by metric
        valid_runs = [run for run in runs if metric_name in run.data.metrics]
        if not valid_runs:
            return None
        
        if mode == "min":
            best_run = min(valid_runs, key=lambda r: r.data.metrics[metric_name])
        else:
            best_run = max(valid_runs, key=lambda r: r.data.metrics[metric_name])
        
        return best_run
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 