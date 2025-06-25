"""
Experiment Tracker - Integration Layer

Integrates MLflow tracking with existing experiment workflow
while maintaining backward compatibility and modularity.
"""

import logging
from typing import Dict, Any, Optional
from .mlflow_client import MLflowClient
import time
import traceback

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """High-level experiment tracking that integrates with existing workflow"""
    
    def __init__(self, config_path: str = "config/mlflow_config.yaml"):
        """Initialize experiment tracker"""
        self.mlflow_client = None
        self.current_run = None
        self.experiment_start_time = None
        
        # Try to initialize MLflow, but don't fail if it's not available
        try:
            self.mlflow_client = MLflowClient(config_path)
            self.enabled = True
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow tracking disabled: {e}")
            self.enabled = False
    
    def start_experiment(self, experiment_name: str, config: Dict[str, Any]) -> bool:
        """Start tracking an experiment"""
        if not self.enabled:
            return False
        
        try:
            self.experiment_start_time = time.time()
            
            # Create run name with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"{experiment_name}_{timestamp}"
            
            # Start MLflow run
            self.current_run = self.mlflow_client.start_run(
                run_name=run_name,
                tags={
                    "experiment_name": experiment_name,
                    "model_type": config.get('model_type', 'unknown'),
                    "target": config.get('target', 'unknown'),
                    "covariates": str(config.get('covariates', 'unknown'))
                }
            )
            
            # Log experiment configuration
            self.mlflow_client.log_experiment_config(config)
            
            logger.info(f"Started tracking experiment: {experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment tracking: {e}")
            return False
    
    def log_data_stage(self, data, stage_name: str, metadata: Dict[str, Any] = None):
        """Log data at different pipeline stages"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            # Log data quality metrics
            self.mlflow_client.log_data_quality(data, stage_name)
            
            # Log stage-specific metadata
            if metadata:
                for key, value in metadata.items():
                    self.mlflow_client.mlflow.log_param(f"{stage_name}_{key}", value)
            
            logger.debug(f"Logged data stage: {stage_name}")
            
        except Exception as e:
            logger.error(f"Failed to log data stage {stage_name}: {e}")
    
    def log_transformations(self, transformation_manager):
        """Log transformation information from TransformationManager"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            # Log transformation metadata
            if hasattr(transformation_manager, 'fitted_transformers'):
                transform_info = {}
                for col, transformer in transformation_manager.fitted_transformers.items():
                    transform_info[col] = transformer.__class__.__name__
                
                self.mlflow_client.log_transformation_info(transform_info)
            
            # Log transformation settings
            settings = {
                'auto_reverse_evaluation': transformation_manager.auto_reverse_for_evaluation,
                'auto_reverse_visualization': transformation_manager.auto_reverse_for_visualization
            }
            
            for key, value in settings.items():
                self.mlflow_client.mlflow.log_param(f"transform_{key}", value)
            
            logger.debug("Logged transformation information")
            
        except Exception as e:
            logger.error(f"Failed to log transformations: {e}")
    
    def log_training_results(self, training_results: Dict[str, Any]):
        """Log model training results"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            # Log model type and configuration
            model_type = training_results.get('model_type', 'unknown')
            self.mlflow_client.mlflow.log_param('model_type', model_type)
            
            # Log quantiles
            quantiles = training_results.get('quantiles', [])
            self.mlflow_client.mlflow.log_param('quantiles', str(quantiles))
            
            # Log feature names
            feature_names = training_results.get('feature_names', [])
            self.mlflow_client.mlflow.log_param('feature_names', str(feature_names))
            
            # Log feature importance if available
            if 'feature_importance' in training_results:
                importance_dict = training_results['feature_importance']
                # Average importance across quantiles
                avg_importance = {}
                for feature in feature_names:
                    importances = [imp_dict.get(feature, 0) for imp_dict in importance_dict.values()]
                    avg_importance[feature] = sum(importances) / len(importances) if importances else 0
                
                self.mlflow_client.log_feature_importance(avg_importance)
            
            # Log model artifacts
            if 'models' in training_results:
                self.mlflow_client.log_model_artifacts(
                    training_results['models'], 
                    model_type
                )
            
            logger.debug("Logged training results")
            
        except Exception as e:
            logger.error(f"Failed to log training results: {e}")
    
    def log_evaluation_results(self, evaluation_results: Dict[str, Any], 
                              training_results: Dict[str, Any] = None):
        """Log model evaluation results"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            # Log performance metrics
            self.mlflow_client.log_model_performance(evaluation_results)
            
            # Log quantile-specific predictions if available
            if 'predictions_test' in evaluation_results and training_results:
                actual = evaluation_results.get('y_test_actual')
                predictions = evaluation_results.get('predictions_test')
                
                if actual is not None and predictions is not None:
                    self.mlflow_client.log_quantile_predictions(
                        predictions, actual, prefix="test"
                    )
                
                # Same for training predictions
                if 'predictions_train' in training_results:
                    actual_train = evaluation_results.get('y_train_actual')
                    predictions_train = training_results.get('predictions_train')
                    
                    if actual_train is not None and predictions_train is not None:
                        self.mlflow_client.log_quantile_predictions(
                            predictions_train, actual_train, prefix="train"
                        )
            
            logger.debug("Logged evaluation results")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation results: {e}")
    
    def log_visualizations(self, visualizer, results: Dict[str, Any]):
        """Log visualizations from the visualizer"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            # This would need to be integrated with your existing visualizer
            # For now, we'll log basic plotting information
            
            self.mlflow_client.mlflow.log_param('visualizations_generated', True)
            
            # If visualizations are saved to files, log them as artifacts
            # This would need to be coordinated with your visualizer module
            
            logger.debug("Logged visualization information")
            
        except Exception as e:
            logger.error(f"Failed to log visualizations: {e}")
    
    def log_experiment_summary(self, summary: Dict[str, Any]):
        """Log final experiment summary"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            # Log duration
            if self.experiment_start_time:
                duration = time.time() - self.experiment_start_time
                self.mlflow_client.mlflow.log_metric('experiment_duration_seconds', duration)
                self.mlflow_client.mlflow.log_metric('experiment_duration_minutes', duration / 60)
            
            # Log summary metrics
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    self.mlflow_client.mlflow.log_metric(f"summary_{key}", value)
                else:
                    self.mlflow_client.mlflow.log_param(f"summary_{key}", str(value))
            
            logger.info("Logged experiment summary")
            
        except Exception as e:
            logger.error(f"Failed to log experiment summary: {e}")
    
    def log_error(self, error: Exception, stage: str = "unknown"):
        """Log experiment errors"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            error_info = {
                'error_stage': stage,
                'error_type': type(error).__name__,
                'error_message': str(error)
            }
            
            for key, value in error_info.items():
                self.mlflow_client.mlflow.log_param(key, value)
            
            # Log traceback as text artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(traceback.format_exc())
                self.mlflow_client.mlflow.log_artifact(f.name, "errors")
            
            logger.error(f"Logged experiment error: {error}")
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def end_experiment(self, success: bool = True):
        """End experiment tracking"""
        if not self.enabled or not self.current_run:
            return
        
        try:
            # Log final status
            self.mlflow_client.mlflow.log_param('experiment_success', success)
            
            # End MLflow run
            self.mlflow_client.end_run()
            self.current_run = None
            
            logger.info(f"Ended experiment tracking (success: {success})")
            
        except Exception as e:
            logger.error(f"Failed to end experiment tracking: {e}")
    
    def is_enabled(self) -> bool:
        """Check if tracking is enabled"""
        return self.enabled
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current run"""
        if not self.enabled or not self.current_run:
            return None
        
        return {
            'run_id': self.current_run.info.run_id,
            'experiment_id': self.current_run.info.experiment_id,
            'start_time': self.current_run.info.start_time,
            'status': self.current_run.info.status
        }
    
    def search_experiments(self, filter_string: str = "") -> list:
        """Search previous experiments"""
        if not self.enabled:
            return []
        
        try:
            return self.mlflow_client.search_runs(filter_string)
        except Exception as e:
            logger.error(f"Failed to search experiments: {e}")
            return [] 