#!/usr/bin/env python3
"""
MLflow-Integrated Experiment Runner

Demonstrates how to run quantile analysis experiments on your data
with comprehensive MLflow tracking and logging.

Usage:
    python run_with_mlflow.py --experiment nav_baseline
    python run_with_mlflow.py --experiment disbursement_baseline --quick
    python run_with_mlflow.py --hyperopt --max_evals 20
"""

import argparse
import sys
import os
from datetime import datetime
import time

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from model_trainer import QuantileModelTrainer
from evaluator import QuantileEvaluator
from visualizer import QuantileVisualizer
from utils import load_config
from tracking import ExperimentTracker, HyperoptRunner, MLflowClient
import mlflow
import yaml

class MLflowExperimentRunner:
    """Experiment runner with full MLflow integration"""
    
    def __init__(self, experiment_name=None, config_dir='../config'):
        """Initialize MLflow experiment runner"""
        self.experiment_name = experiment_name
        self.config_dir = config_dir
        
        # Load configurations
        self.load_configurations()
        
        # Initialize MLflow tracker
        self.tracker = ExperimentTracker()
        
        print(f"🚀 MLflow Experiment Runner Initialized")
        print(f"Tracking enabled: {self.tracker.is_enabled()}")
        if experiment_name:
            print(f"Experiment: {experiment_name}")
        
    def load_configurations(self):
        """Load all configuration files"""
        self.data_config = load_config(os.path.join(self.config_dir, 'data_config.yaml'))
        self.model_config = load_config(os.path.join(self.config_dir, 'model_config.yaml'))
        self.experiment_config = load_config(os.path.join(self.config_dir, 'experiment_config.yaml'))
        
        # Load MLflow config if available
        try:
            self.mlflow_config = load_config(os.path.join(self.config_dir, 'mlflow_config.yaml'))
        except:
            self.mlflow_config = {}
        
        print("✅ Loaded all configuration files")
    
    def run_single_experiment(self, experiment_name: str, save_visualizations: bool = True):
        """Run a single experiment with MLflow tracking"""
        
        if experiment_name not in self.experiment_config['experiments']:
            raise ValueError(f"Experiment '{experiment_name}' not found in configuration")
        
        exp_config = self.experiment_config['experiments'][experiment_name]
        
        print(f"\n🔬 Running Experiment: {experiment_name}")
        print("=" * 60)
        print(f"Target: {exp_config['target']}")
        print(f"Covariates: {exp_config['covariates']}")
        print(f"Model type: {exp_config['model_type']}")
        print("=" * 60)
        
        # Prepare experiment configuration for tracking
        full_config = {
            'experiment_name': experiment_name,
            'target': exp_config['target'],
            'model_type': exp_config['model_type'],
            'covariates': exp_config['covariates'],
            'quantiles': exp_config.get('quantiles', 'default'),
            'data_config': self.data_config,
            'model_config': self.model_config[exp_config['model_type']],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Start MLflow tracking
            success = self.tracker.start_experiment(experiment_name, full_config)
            if success:
                print("📊 Started MLflow tracking")
            
            # Step 1: Load and prepare data
            print("\n📂 Step 1: Loading and preparing data...")
            data_loader = DataLoader(self.data_config, self.experiment_config)
            data_dict = data_loader.prepare_experiment_data(experiment_name)
            
            # Log data stage
            self.tracker.log_data_stage(
                data_dict['df_full'], 
                "raw_data",
                {
                    'file_path': self.data_config['file_path'],
                    'train_samples': len(data_dict['X_train']),
                    'test_samples': len(data_dict['X_test']),
                    'features': len(data_dict['feature_names'])
                }
            )
            
            # Log transformations
            transformation_manager = data_loader.get_transformation_manager()
            self.tracker.log_transformations(transformation_manager)
            
            print(f"✅ Data loaded: {len(data_dict['df_full'])} samples, {len(data_dict['feature_names'])} features")
            
            # Step 2: Train models
            print("\n🤖 Step 2: Training quantile models...")
            trainer = QuantileModelTrainer(self.model_config)
            
            training_results = trainer.train_quantile_models(
                data_dict['X_train'], 
                data_dict['y_train'], 
                exp_config['model_type'], 
                data_dict['feature_names'],
                exp_config.get('quantiles', 'default')
            )
            
            # Log training results
            self.tracker.log_training_results(training_results)
            
            print(f"✅ Models trained for {len(training_results['quantiles'])} quantiles")
            
            # Step 3: Evaluate models
            print("\n📊 Step 3: Evaluating models...")
            evaluator = QuantileEvaluator(
                data_config=self.data_config, 
                transformation_manager=transformation_manager
            )
            
            target_name = exp_config['target']
            target_config = self.experiment_config['targets'][target_name]
            target_column = target_config['column']
            
            evaluation_results = evaluator.evaluate_models(
                training_results, data_dict, target_column
            )
            
            # Log evaluation results
            self.tracker.log_evaluation_results(evaluation_results, training_results)
            
            print("✅ Model evaluation completed")
            
            # Step 4: Create visualizations (optional)
            if save_visualizations:
                print("\n📈 Step 4: Creating visualizations...")
                # Note: For MLflow demo, we'll skip heavy visualizations to keep it fast
                # In production, you'd want to save and log plots as artifacts
                print("⏭️  Skipping detailed visualizations for demo speed")
            
            # Generate summary
            summary = evaluator.generate_evaluation_summary(evaluation_results, training_results)
            self.tracker.log_experiment_summary(summary)
            
            # Log success
            self.tracker.end_experiment(success=True)
            
            print(f"\n🎉 Experiment '{experiment_name}' completed successfully!")
            print("📊 View results in MLflow UI: http://localhost:5000")
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'summary': summary,
                'mlflow_run': self.tracker.get_run_info()
            }
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {str(e)}")
            self.tracker.log_error(e, "experiment_execution")
            self.tracker.end_experiment(success=False)
            raise
    
    def run_quick_comparison(self, experiment_names: list):
        """Run multiple experiments for comparison"""
        
        print(f"\n🔬 Running Quick Comparison: {experiment_names}")
        print("=" * 60)
        
        results = {}
        
        for exp_name in experiment_names:
            print(f"\n▶️  Running {exp_name}...")
            try:
                result = self.run_single_experiment(exp_name, save_visualizations=False)
                results[exp_name] = result
                
                # Print quick summary
                summary = result['summary']
                print(f"✅ {exp_name}: Avg coverage = {summary.get('average_coverage', 'N/A'):.3f}")
                
            except Exception as e:
                print(f"❌ {exp_name} failed: {str(e)}")
                results[exp_name] = {'error': str(e)}
        
        # Create comparison summary in MLflow
        if len(results) > 1:
            self._log_comparison_summary(results)
        
        return results
    
    def _log_comparison_summary(self, results: dict):
        """Log comparison summary to MLflow"""
        try:
            mlflow.set_experiment("experiment_comparison")
            
            with mlflow.start_run(run_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                comparison_data = {}
                
                for exp_name, result in results.items():
                    if 'error' not in result:
                        summary = result['summary']
                        comparison_data[f"{exp_name}_coverage"] = summary.get('average_coverage', 0)
                        comparison_data[f"{exp_name}_pinball_loss"] = summary.get('average_pinball_loss', 0)
                
                # Log comparison metrics
                for metric, value in comparison_data.items():
                    mlflow.log_metric(metric, value)
                
                # Log experiment names
                mlflow.log_param("experiments_compared", list(results.keys()))
                mlflow.log_param("comparison_timestamp", datetime.now().isoformat())
                
                print("📊 Comparison summary logged to MLflow")
                
        except Exception as e:
            print(f"⚠️  Failed to log comparison: {e}")
    
    def run_hyperparameter_optimization(self, base_experiment: str, max_evals: int = 20):
        """Run hyperparameter optimization with MLflow tracking"""
        
        print(f"\n🔧 Running Hyperparameter Optimization")
        print(f"Base experiment: {base_experiment}")
        print(f"Max evaluations: {max_evals}")
        print("=" * 60)
        
        if base_experiment not in self.experiment_config['experiments']:
            raise ValueError(f"Base experiment '{base_experiment}' not found")
        
        try:
            # Initialize hyperopt runner
            mlflow_client = MLflowClient()
            hyperopt_runner = HyperoptRunner(mlflow_client)
            
            # Prepare data once
            print("📂 Preparing data...")
            data_loader = DataLoader(self.data_config, self.experiment_config)
            data_dict = data_loader.prepare_experiment_data(base_experiment)
            
            exp_config = self.experiment_config['experiments'][base_experiment]
            
            # Define objective function
            def objective(params):
                """Objective function for hyperparameter optimization"""
                try:
                    # Update model config with suggested parameters
                    model_type = exp_config['model_type']
                    updated_model_config = self.model_config.copy()
                    updated_model_config[model_type].update(params)
                    
                    # Train model with suggested parameters
                    trainer = QuantileModelTrainer(updated_model_config)
                    training_results = trainer.train_quantile_models(
                        data_dict['X_train'], 
                        data_dict['y_train'], 
                        model_type, 
                        data_dict['feature_names'],
                        exp_config.get('quantiles', 'default')
                    )
                    
                    # Quick evaluation (just pinball loss)
                    evaluator = QuantileEvaluator(self.data_config)
                    target_name = exp_config['target']
                    target_config = self.experiment_config['targets'][target_name]
                    target_column = target_config['column']
                    
                    evaluation_results = evaluator.evaluate_models(
                        training_results, data_dict, target_column
                    )
                    
                    # Return average pinball loss (to minimize)
                    return evaluation_results['average_pinball_loss']
                    
                except Exception as e:
                    print(f"⚠️  Trial failed: {e}")
                    return float('inf')  # Return worst possible score
            
            # Run optimization
            best_params = hyperopt_runner.optimize_hyperparameters(
                objective_func=objective,
                model_type=exp_config['model_type'],
                max_evals=max_evals,
                experiment_name=f"{base_experiment}_hyperopt"
            )
            
            print(f"\n🏆 Best parameters found:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            
            print("📊 View optimization results in MLflow UI: http://localhost:5000")
            
            return best_params
            
        except Exception as e:
            print(f"❌ Hyperparameter optimization failed: {str(e)}")
            raise

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run quantile analysis experiments with MLflow tracking')
    
    # Experiment options
    parser.add_argument('--experiment', type=str, 
                       help='Single experiment to run (e.g., nav_baseline)')
    parser.add_argument('--compare', nargs='+', 
                       help='Multiple experiments to compare (e.g., nav_baseline disbursement_baseline)')
    parser.add_argument('--hyperopt', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--base_experiment', type=str, default='nav_baseline',
                       help='Base experiment for hyperparameter optimization')
    
    # Options
    parser.add_argument('--max_evals', type=int, default=20,
                       help='Maximum evaluations for hyperparameter optimization')
    parser.add_argument('--quick', action='store_true',
                       help='Skip visualizations for faster execution')
    parser.add_argument('--config_dir', type=str, default='config',
                       help='Configuration directory path')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = MLflowExperimentRunner(config_dir=args.config_dir)
    
    try:
        if args.hyperopt:
            # Run hyperparameter optimization
            runner.run_hyperparameter_optimization(
                base_experiment=args.base_experiment,
                max_evals=args.max_evals
            )
            
        elif args.compare:
            # Run comparison of multiple experiments
            runner.run_quick_comparison(args.compare)
            
        elif args.experiment:
            # Run single experiment
            runner.run_single_experiment(
                args.experiment, 
                save_visualizations=not args.quick
            )
            
        else:
            # Default: run a quick demo
            print("🎯 Running default demo...")
            print("Available experiments:")
            experiments = list(runner.experiment_config['experiments'].keys())
            for i, exp in enumerate(experiments, 1):
                print(f"  {i}. {exp}")
            
            # Run the first two experiments for comparison
            if len(experiments) >= 2:
                runner.run_quick_comparison(experiments[:2])
            else:
                runner.run_single_experiment(experiments[0], save_visualizations=False)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    print("\n✅ MLflow experiment runner completed")
    print("📊 View all results at: http://localhost:5000")
    return 0

if __name__ == "__main__":
    exit(main())