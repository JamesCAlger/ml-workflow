#!/usr/bin/env python3
"""
MLflow Hyperparameter Optimization Demonstration

This script demonstrates how to use the new MLflow integration
for hyperparameter tuning in your quantile analysis platform.
"""

import argparse
import sys
sys.path.append('src')

from tracking import MLflowClient, HyperoptRunner, ExperimentTracker
from data_loader import DataLoader
from transformation_manager import TransformationManager
from model_trainer import QuantileModelTrainer
from evaluator import QuantileEvaluator
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_dir: str = "config"):
    """Load all configuration files"""
    configs = {}
    
    config_files = [
        'data_config.yaml',
        'experiment_config.yaml', 
        'model_config.yaml',
        'mlflow_config.yaml'
    ]
    
    for config_file in config_files:
        try:
            with open(f"{config_dir}/{config_file}", 'r') as f:
                config_name = config_file.replace('.yaml', '')
                configs[config_name] = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found")
            
    return configs


def prepare_data_for_optimization(target: str, covariates: str, configs: dict):
    """Prepare data for hyperparameter optimization"""
    
    # Load data
    data_loader = DataLoader(configs['data_config'])
    data = data_loader.load_data()
    
    # Get target and covariate configuration
    target_config = configs['experiment_config']['targets'][target]
    covariate_config = configs['experiment_config']['covariates'][covariates]
    
    # Setup transformations
    transformation_manager = TransformationManager()
    
    # Apply transformations based on target
    target_column = target_config['column']
    if 'log' in target_column and target_column not in data.columns:
        if 'disbursement' in target_column:
            base_col = 'disbursement'
        elif 'nav' in target_column:
            base_col = 'nav'
        else:
            base_col = target_column.replace('_log', '')
            
        transformation_manager.fit_transform(data, {base_col: 'log_transform'})
    
    # Prepare features and target
    feature_names = covariate_config['features']
    X = data[feature_names].copy()
    y = data[target_column].copy()
    
    # Remove missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Prepared data: {len(X)} samples, {len(feature_names)} features")
    logger.info(f"Target: {target_column}, Features: {feature_names}")
    
    return {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'target_column': target_column,
        'transformation_manager': transformation_manager
    }


def create_training_function(configs: dict):
    """Create training function for hyperparameter optimization"""
    
    def train_model(data_dict, model_params):
        """Training function for optimization"""
        
        # Initialize model trainer with custom parameters
        model_trainer = QuantileModelTrainer(configs['model_config'])
        
        # Get data
        X = data_dict['X']
        y = data_dict['y']
        feature_names = data_dict['feature_names']
        
        # Split data (simple split for demo)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Update model config with suggested parameters
        model_config = configs['model_config'].copy()
        
        # Determine model type from parameters
        if 'max_depth' in model_params and 'n_estimators' in model_params:
            model_type = 'xgboost'
            model_config['xgboost'].update(model_params)
        elif 'num_leaves' in model_params:
            model_type = 'lightgbm'
            model_config['lightgbm'].update(model_params)
        else:
            model_type = 'lightgbm'  # default
            
        # Train models
        quantiles = [0.1, 0.5, 0.9]
        
        if model_type == 'xgboost':
            models, predictions_train, feature_importance = model_trainer.train_xgboost_quantiles(
                X_train, y_train, quantiles, feature_names
            )
        else:
            models, predictions_train, feature_importance = model_trainer.train_lightgbm_quantiles(
                X_train, y_train, quantiles, feature_names
            )
        
        # Make test predictions
        predictions_test = {}
        for quantile, model in models.items():
            if model_type == 'xgboost':
                predictions_test[quantile] = model.predict(X_test)
            else:
                predictions_test[quantile] = model.predict(X_test, num_iteration=model.best_iteration)
        
        return {
            'models': models,
            'predictions_train': predictions_train,
            'predictions_test': predictions_test,
            'feature_importance': feature_importance,
            'model_type': model_type,
            'quantiles': quantiles,
            'feature_names': feature_names,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
    
    return train_model


def create_evaluation_function(configs: dict):
    """Create evaluation function for hyperparameter optimization"""
    
    def evaluate_model(training_results, data_dict):
        """Evaluation function for optimization"""
        
        evaluator = QuantileEvaluator()
        
        # Extract results
        predictions_test = training_results['predictions_test']
        y_test = training_results['y_test']
        quantiles = training_results['quantiles']
        
        # Import pinball_loss function
        from utils import pinball_loss
        
        # Calculate pinball losses
        pinball_losses = {}
        for quantile in quantiles:
            pred = predictions_test[quantile]
            loss = pinball_loss(y_test, pred, quantile)
            pinball_losses[f'pinball_loss_q{int(quantile*100)}'] = loss
        
        # Calculate average pinball loss
        avg_pinball_loss = sum(pinball_losses.values()) / len(pinball_losses)
        pinball_losses['test_pinball_loss_avg'] = avg_pinball_loss
        
        # Add simple coverage for 80% interval if available
        coverage_results = {}
        if 0.1 in predictions_test and 0.9 in predictions_test:
            from utils import empirical_coverage
            coverage = empirical_coverage(y_test, predictions_test[0.1], predictions_test[0.9])
            coverage_results['coverage_80pct'] = coverage
        
        # Combine all metrics
        evaluation_results = {
            **pinball_losses,
            **coverage_results,
            'predictions_test': predictions_test,
            'y_test_actual': y_test
        }
        
        return evaluation_results
    
    return evaluate_model


def run_hyperparameter_optimization(model_type: str, target: str, covariates: str, 
                                   max_trials: int = 20, config_dir: str = "config"):
    """Run hyperparameter optimization with MLflow tracking"""
    
    logger.info(f"Starting hyperparameter optimization for {model_type}")
    logger.info(f"Target: {target}, Covariates: {covariates}")
    
    # Load configurations
    configs = load_config(config_dir)
    
    # Update MLflow config for shorter demo
    if 'mlflow_config' in configs:
        configs['mlflow_config']['mlflow']['hyperopt']['max_evals'] = max_trials
    
    # Initialize MLflow components
    mlflow_client = MLflowClient(f"{config_dir}/mlflow_config.yaml")
    hyperopt_runner = HyperoptRunner(mlflow_client)
    
    # Prepare data
    data_dict = prepare_data_for_optimization(target, covariates, configs)
    
    # Create training and evaluation functions
    train_func = create_training_function(configs)
    evaluate_func = create_evaluation_function(configs)
    
    # Run optimization
    study = hyperopt_runner.run_optimization(
        train_func=train_func,
        evaluate_func=evaluate_func,
        model_type=model_type,
        data_dict=data_dict,
        target_metric='test_pinball_loss_avg',
        direction='minimize'
    )
    
    # Get best parameters
    best_params = hyperopt_runner.get_best_params(study)
    
    logger.info("Optimization completed!")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best value: {study.best_trial.value}")
    logger.info(f"Total trials: {len(study.trials)}")
    
    return study, best_params


def run_comparison_optimization(targets: list, model_types: list, 
                              max_trials: int = 10, config_dir: str = "config"):
    """Run optimization comparison across multiple targets and models"""
    
    logger.info("Starting comparison optimization")
    
    results = {}
    configs = load_config(config_dir)
    
    # Initialize components
    mlflow_client = MLflowClient(f"{config_dir}/mlflow_config.yaml")
    hyperopt_runner = HyperoptRunner(mlflow_client)
    
    for target in targets:
        results[target] = {}
        
        # Prepare data for this target
        data_dict = prepare_data_for_optimization(target, 'base_set', configs)
        
        # Create functions
        train_func = create_training_function(configs)
        evaluate_func = create_evaluation_function(configs)
        
        for model_type in model_types:
            logger.info(f"Optimizing {model_type} for {target}")
            
            # Update config for shorter demo
            if 'mlflow_config' in configs:
                configs['mlflow_config']['mlflow']['hyperopt']['max_evals'] = max_trials
            
            study = hyperopt_runner.run_optimization(
                train_func=train_func,
                evaluate_func=evaluate_func,
                model_type=model_type,
                data_dict=data_dict,
                study_name=f"{target}_{model_type}_comparison",
                target_metric='test_pinball_loss_avg',
                direction='minimize'
            )
            
            results[target][model_type] = {
                'best_value': study.best_trial.value,
                'best_params': study.best_trial.params,
                'n_trials': len(study.trials)
            }
    
    # Create comparison report
    report = hyperopt_runner.create_optimization_report({
        f"{target}_{model_type}": study 
        for target in targets 
        for model_type in model_types
    })
    
    logger.info("Comparison optimization completed!")
    logger.info(f"Results: {results}")
    
    return results, report


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='MLflow Hyperparameter Optimization Demo')
    parser.add_argument('--model_type', choices=['lightgbm', 'xgboost'], 
                       default='lightgbm', help='Model type to optimize')
    parser.add_argument('--target', default='nav_log', 
                       help='Target variable to predict')
    parser.add_argument('--covariates', default='base_set',
                       help='Covariate set to use')
    parser.add_argument('--max_trials', type=int, default=20,
                       help='Maximum number of optimization trials')
    parser.add_argument('--config_dir', default='config',
                       help='Configuration directory')
    parser.add_argument('--comparison', action='store_true',
                       help='Run comparison across multiple models/targets')
    
    args = parser.parse_args()
    
    print("🚀 MLflow Hyperparameter Optimization Demo")
    print("=" * 50)
    
    if args.comparison:
        # Run comparison optimization
        targets = ['nav_log', 'disbursement_log']
        model_types = ['lightgbm', 'xgboost']
        
        results, report = run_comparison_optimization(
            targets=targets,
            model_types=model_types,
            max_trials=args.max_trials,
            config_dir=args.config_dir
        )
        
        print("\n📊 Comparison Results:")
        for target, model_results in results.items():
            print(f"\n{target}:")
            for model_type, metrics in model_results.items():
                print(f"  {model_type}: {metrics['best_value']:.2f} "
                      f"(trials: {metrics['n_trials']})")
    
    else:
        # Run single optimization
        study, best_params = run_hyperparameter_optimization(
            model_type=args.model_type,
            target=args.target,
            covariates=args.covariates,
            max_trials=args.max_trials,
            config_dir=args.config_dir
        )
        
        print(f"\n🎯 Best Parameters for {args.model_type}:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    
    print("\n✅ Demo completed!")
    print("🌐 View results at: http://localhost:5000")
    print("💡 Start MLflow UI with: mlflow ui --port 5000")


if __name__ == "__main__":
    main() 