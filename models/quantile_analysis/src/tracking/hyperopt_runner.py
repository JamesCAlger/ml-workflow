"""
Hyperparameter Optimization with MLflow Integration

Provides automated hyperparameter tuning using Optuna or Hyperopt
with full MLflow tracking of all trials.
"""

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import yaml
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class HyperoptRunner:
    """Hyperparameter optimization with MLflow tracking"""
    
    def __init__(self, mlflow_client, config_path: str = "config/mlflow_config.yaml"):
        """Initialize hyperparameter optimizer"""
        self.mlflow_client = mlflow_client
        self.config = self._load_config(config_path)
        self.hyperopt_config = self.config.get('mlflow', {}).get('hyperopt', {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}
    
    def define_search_space(self, model_type: str) -> Dict[str, Any]:
        """Define hyperparameter search space for different model types"""
        
        if model_type == 'lightgbm':
            return {
                'num_leaves': ('int', 10, 100),
                'learning_rate': ('float', 0.01, 0.3),
                'feature_fraction': ('float', 0.5, 1.0),
                'bagging_fraction': ('float', 0.5, 1.0),
                'bagging_freq': ('int', 1, 10),
                'num_boost_round': ('int', 50, 500),
                'max_depth': ('int', 3, 15),
                'min_data_in_leaf': ('int', 10, 100)
            }
        
        elif model_type == 'xgboost':
            return {
                'max_depth': ('int', 3, 15),
                'learning_rate': ('float', 0.01, 0.3),
                'subsample': ('float', 0.5, 1.0),
                'colsample_bytree': ('float', 0.5, 1.0),
                'n_estimators': ('int', 50, 500),
                'min_child_weight': ('int', 1, 10),
                'gamma': ('float', 0, 1),
                'reg_alpha': ('float', 0, 1),
                'reg_lambda': ('float', 0, 1)
            }
        
        elif model_type == 'sklearn_quantile':
            return {
                'alpha': ('float', 0.001, 10.0),
                'fit_intercept': ('categorical', [True, False]),
                'solver': ('categorical', ['highs', 'interior-point'])
            }
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def suggest_params(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest hyperparameters based on search space"""
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config[0]
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])
            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_loguniform(param_name, param_config[1], param_config[2])
        
        return params
    
    def objective_function(self, trial: optuna.Trial, 
                          train_func: Callable,
                          evaluate_func: Callable,
                          search_space: Dict[str, Any],
                          data_dict: Dict[str, Any],
                          target_metric: str = 'test_pinball_loss_avg') -> float:
        """Objective function for optimization"""
        
        # Suggest hyperparameters
        suggested_params = self.suggest_params(trial, search_space)
        
        # Start MLflow run for this trial
        with self.mlflow_client.start_run(
            run_name=f"trial_{trial.number}",
            tags={
                "trial_number": str(trial.number),
                "optimization": "optuna"
            }
        ):
            try:
                # Log trial parameters
                for param_name, param_value in suggested_params.items():
                    mlflow.log_param(f"trial_{param_name}", param_value)
                
                # Train model with suggested parameters
                training_results = train_func(
                    data_dict, 
                    model_params=suggested_params
                )
                
                # Evaluate model
                evaluation_results = evaluate_func(
                    training_results,
                    data_dict
                )
                
                # Log evaluation metrics
                self.mlflow_client.log_model_performance(evaluation_results)
                
                # Get target metric for optimization
                target_value = evaluation_results.get(target_metric, float('inf'))
                
                # Log the target metric
                mlflow.log_metric("optimization_target", target_value)
                
                logger.info(f"Trial {trial.number}: {target_metric} = {target_value}")
                
                return target_value
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                # Log the error
                mlflow.log_param("error", str(e))
                return float('inf')
    
    def run_optimization(self, 
                        train_func: Callable,
                        evaluate_func: Callable,
                        model_type: str,
                        data_dict: Dict[str, Any],
                        study_name: str = None,
                        target_metric: str = 'test_pinball_loss_avg',
                        direction: str = 'minimize') -> optuna.Study:
        """Run hyperparameter optimization"""
        
        # Setup study name
        if study_name is None:
            study_name = f"{model_type}_optimization_{int(time.time())}"
        
        # Get search space
        search_space = self.define_search_space(model_type)
        
        # Create Optuna study
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=None  # In-memory storage
        )
        
        # Setup MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.mlflow_client.config['mlflow']['tracking_uri'],
            metric_name=target_metric
        )
        
        # Optimization parameters
        max_evals = self.hyperopt_config.get('max_evals', 50)
        timeout = self.hyperopt_config.get('timeout', 3600)
        
        logger.info(f"Starting optimization with {max_evals} trials")
        logger.info(f"Search space: {search_space}")
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective_function(
                trial, train_func, evaluate_func, 
                search_space, data_dict, target_metric
            ),
            n_trials=max_evals,
            timeout=timeout,
            callbacks=[mlflow_callback]
        )
        
        # Log best results
        self._log_optimization_results(study, model_type)
        
        return study
    
    def _log_optimization_results(self, study: optuna.Study, model_type: str):
        """Log optimization results to MLflow"""
        
        # Start a summary run
        with self.mlflow_client.start_run(
            run_name=f"{model_type}_optimization_summary",
            tags={"optimization": "summary", "model_type": model_type}
        ):
            # Log best trial information
            best_trial = study.best_trial
            mlflow.log_metric("best_value", best_trial.value)
            mlflow.log_metric("n_trials", len(study.trials))
            
            # Log best parameters
            for param_name, param_value in best_trial.params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            # Log optimization statistics
            values = [trial.value for trial in study.trials if trial.value != float('inf')]
            if values:
                mlflow.log_metric("optimization_mean", np.mean(values))
                mlflow.log_metric("optimization_std", np.std(values))
                mlflow.log_metric("optimization_min", np.min(values))
                mlflow.log_metric("optimization_max", np.max(values))
            
            # Create and log optimization history plot
            try:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Optimization history
                ax1.plot([trial.value for trial in study.trials])
                ax1.set_xlabel('Trial')
                ax1.set_ylabel('Objective Value')
                ax1.set_title('Optimization History')
                ax1.grid(True)
                
                # Parameter importance (if available)
                if len(study.trials) > 10:
                    importance = optuna.importance.get_param_importances(study)
                    params = list(importance.keys())
                    importances = list(importance.values())
                    
                    ax2.barh(params, importances)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance')
                
                plt.tight_layout()
                self.mlflow_client.log_visualization(fig, "optimization_results.png")
                plt.close()
                
            except Exception as e:
                logger.warning(f"Failed to create optimization plots: {e}")
        
        logger.info(f"Optimization completed. Best value: {best_trial.value}")
        logger.info(f"Best parameters: {best_trial.params}")
    
    def run_parallel_optimization(self,
                                 train_func: Callable,
                                 evaluate_func: Callable,
                                 model_types: List[str],
                                 data_dict: Dict[str, Any],
                                 max_workers: int = 3) -> Dict[str, optuna.Study]:
        """Run optimization for multiple model types in parallel"""
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for model_type in model_types:
                future = executor.submit(
                    self.run_optimization,
                    train_func,
                    evaluate_func,
                    model_type,
                    data_dict,
                    study_name=f"{model_type}_parallel_opt"
                )
                futures[model_type] = future
            
            # Collect results
            for model_type, future in futures.items():
                try:
                    results[model_type] = future.result()
                    logger.info(f"Completed optimization for {model_type}")
                except Exception as e:
                    logger.error(f"Optimization failed for {model_type}: {e}")
        
        return results
    
    def get_best_params(self, study: optuna.Study) -> Dict[str, Any]:
        """Get best parameters from study"""
        return study.best_trial.params
    
    def create_optimization_report(self, studies: Dict[str, optuna.Study]) -> Dict[str, Any]:
        """Create a comprehensive optimization report"""
        report = {
            'summary': {},
            'model_comparison': {},
            'best_models': {}
        }
        
        for model_type, study in studies.items():
            best_trial = study.best_trial
            
            report['best_models'][model_type] = {
                'best_value': best_trial.value,
                'best_params': best_trial.params,
                'n_trials': len(study.trials)
            }
            
            # Calculate statistics
            values = [trial.value for trial in study.trials if trial.value != float('inf')]
            if values:
                report['summary'][model_type] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'improvement': (np.max(values) - np.min(values)) / np.max(values) * 100
                }
        
        # Find overall best model
        if report['best_models']:
            best_model = min(report['best_models'].items(), 
                           key=lambda x: x[1]['best_value'])
            report['overall_best'] = {
                'model_type': best_model[0],
                'value': best_model[1]['best_value'],
                'params': best_model[1]['best_params']
            }
        
        return report 