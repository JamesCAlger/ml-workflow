#!/usr/bin/env python3
"""
Main experiment runner for quantile analysis

Usage:
    python run_experiment.py --experiment baseline
    python run_experiment.py --experiment extended_features --save_results
"""

import argparse
import sys
import os
from datetime import datetime

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from model_trainer import QuantileModelTrainer
from evaluator import QuantileEvaluator
from visualizer import QuantileVisualizer
from utils import (load_config, save_results, create_experiment_dir, 
                   save_config_copy, save_predictions, save_models)

class ExperimentRunner:
    """Main experiment runner for quantile analysis"""
    
    def __init__(self, experiment_name, config_dir='../config', results_dir='../results'):
        """Initialize experiment runner"""
        self.experiment_name = experiment_name
        self.config_dir = config_dir
        self.results_dir = results_dir
        
        # Load configurations
        self.load_configurations()
        
        # Validate experiment exists
        if experiment_name not in self.experiment_config['experiments']:
            raise ValueError(f"Experiment '{experiment_name}' not found in configuration")
        
        # Get experiment-specific configuration
        self.exp_config = self.experiment_config['experiments'][experiment_name]
        
        print(f"Initialized experiment runner for: {experiment_name}")
        print(f"Target: {self.exp_config['target']}")
        print(f"Covariates: {self.exp_config['covariates']}")
        print(f"Model type: {self.exp_config['model_type']}")
    
    def load_configurations(self):
        """Load all configuration files"""
        self.data_config = load_config(os.path.join(self.config_dir, 'data_config.yaml'))
        self.model_config = load_config(os.path.join(self.config_dir, 'model_config.yaml'))
        self.experiment_config = load_config(os.path.join(self.config_dir, 'experiment_config.yaml'))
        
        print("Loaded all configuration files")
    
    def setup_experiment_directory(self):
        """Create experiment directory and setup output paths"""
        self.experiment_dir = create_experiment_dir(self.results_dir, self.experiment_name)
        print(f"Created experiment directory: {self.experiment_dir}")
        
        # Save configuration copies
        all_configs = {
            'data_config': self.data_config,
            'model_config': self.model_config,
            'experiment_config': self.experiment_config,
            'experiment_used': self.exp_config
        }
        save_config_copy(all_configs, self.experiment_dir, 'full_config.yaml')
        
        return self.experiment_dir
    
    def load_and_prepare_data(self):
        """Load and prepare data for the experiment"""
        print("="*60)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("="*60)
        
        # Initialize data loader
        data_loader = DataLoader(self.data_config, self.experiment_config)
        
        # Prepare data for this specific experiment
        data_dict = data_loader.prepare_experiment_data(self.experiment_name)
        
        print(f"Data preparation completed for experiment: {self.experiment_name}")
        return data_dict, data_loader
    
    def train_models(self, data_dict):
        """Train quantile models"""
        print("="*60)
        print("STEP 2: MODEL TRAINING")
        print("="*60)
        
        # Initialize model trainer
        trainer = QuantileModelTrainer(self.model_config)
        
        # Get model parameters from experiment config
        model_type = self.exp_config['model_type']
        quantile_set = self.exp_config.get('quantiles', 'default')
        
        # Train models
        training_results = trainer.train_quantile_models(
            data_dict['X_train'], 
            data_dict['y_train'], 
            model_type, 
            data_dict['feature_names'],
            quantile_set
        )
        
        # Save models
        if hasattr(self, 'experiment_dir') and self.experiment_dir:
            trainer.save_trained_models(training_results, self.experiment_dir)
        
        print("Model training completed!")
        return training_results, trainer
    
    def evaluate_models(self, training_results, data_dict, data_loader):
        """Evaluate trained models"""
        print("="*60)
        print("STEP 3: MODEL EVALUATION")
        print("="*60)
        
        # Get transformation manager from data loader
        transformation_manager = data_loader.get_transformation_manager()
        
        # Initialize evaluator with data config and transformation manager
        evaluator = QuantileEvaluator(data_config=self.data_config, 
                                    transformation_manager=transformation_manager)
        
        # Get target column for back-transformation
        target_name = self.exp_config['target']
        target_config = self.experiment_config['targets'][target_name]
        target_column = target_config['column']
        
        # Run comprehensive evaluation with automatic back-transformation
        evaluation_results = evaluator.evaluate_models(training_results, data_dict, target_column)
        
        # Generate evaluation summary
        summary = evaluator.generate_evaluation_summary(evaluation_results, training_results)
        
        print("Model evaluation completed!")
        return evaluation_results, summary, evaluator
    
    def create_visualizations(self, training_results, evaluation_results, data_dict, data_loader):
        """Create all visualizations"""
        print("="*60)
        print("STEP 4: VISUALIZATION")
        print("="*60)
        
        # Get transformation manager from data loader
        transformation_manager = data_loader.get_transformation_manager()
        
        # Initialize visualizer with data config and transformation manager
        visualizer = QuantileVisualizer(self.experiment_dir, 
                                      data_config=self.data_config,
                                      transformation_manager=transformation_manager)
        
        # Create all standard visualizations (predictions are already back-transformed)
        visualizer.create_all_visualizations(
            data_dict['df_full'], 
            evaluation_results, 
            training_results, 
            save=True
        )
        
        print("Visualization completed!")
        return visualizer
    
    def save_experiment_results(self, training_results, evaluation_results, summary, data_dict):
        """Save all experiment results"""
        print("="*60)
        print("STEP 5: SAVING RESULTS")
        print("="*60)
        
        # Save evaluation results
        save_results(evaluation_results, self.experiment_dir, "evaluation_results.json")
        
        # Save summary
        save_results(summary, self.experiment_dir, "experiment_summary.json")
        
        # Save training metadata
        training_metadata = {
            'model_type': training_results['model_type'],
            'quantiles': training_results['quantiles'],
            'feature_names': training_results['feature_names'],
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat()
        }
        save_results(training_metadata, self.experiment_dir, "training_metadata.json")
        
        # Save predictions
        if 'predictions' in evaluation_results:
            predictions = evaluation_results['predictions']
            for pred_type, pred_data in predictions.items():
                if pred_data:
                    save_predictions(pred_data, self.experiment_dir, f"predictions_{pred_type}.csv")
        
        print(f"All results saved to: {self.experiment_dir}")
    
    def print_final_summary(self, summary, training_results):
        """Print final experiment summary"""
        print("="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        model_overview = summary['model_overview']
        performance_metrics = summary['performance_metrics']
        
        print(f"\n📊 EXPERIMENT: {self.experiment_name.upper()}")
        print(f"   • Model type: {model_overview['model_type']}")
        print(f"   • Quantiles trained: {model_overview['n_quantiles']}")
        print(f"   • Features: {model_overview['features']}")
        print(f"   • {model_overview['aggregation_covariate'].title()} categories analyzed: {model_overview['n_covariates']}")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   • Average test pinball loss: {performance_metrics['avg_test_pinball_loss']:.2f}")
        print(f"   • Best coverage interval: {performance_metrics['best_coverage_interval']}")
        
        if summary['coverage_summary']:
            print(f"\n🎯 COVERAGE ANALYSIS:")
            for coverage in summary['coverage_summary']:
                print(f"   • {coverage['Interval']} interval: Expected {coverage['Expected_Coverage']:.1%}, "
                      f"Actual {coverage['Empirical_Coverage']:.1%} "
                      f"(Δ{coverage['Coverage_Difference']:+.3f})")
        
        print(f"\n📁 RESULTS SAVED TO:")
        if hasattr(self, 'experiment_dir') and self.experiment_dir:
            print(f"   {self.experiment_dir}")
        else:
            print(f"   Results not saved (--no_save option used)")
        
        print(f"\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def run_full_experiment(self, save_results=True):
        """Run the complete experiment pipeline"""
        start_time = datetime.now()
        
        print("🚀 Starting Quantile Analysis Experiment")
        print(f"Experiment: {self.experiment_name}")
        print(f"Start time: {start_time}")
        print("="*80)
        
        try:
            # Setup experiment directory
            if save_results:
                self.setup_experiment_directory()
            
            # Step 1: Load and prepare data
            data_dict, data_loader = self.load_and_prepare_data()
            
            # Step 2: Train models
            training_results, trainer = self.train_models(data_dict)
            
            # Step 3: Evaluate models
            evaluation_results, summary, evaluator = self.evaluate_models(training_results, data_dict, data_loader)
            
            # Step 4: Create visualizations
            if save_results:
                visualizer = self.create_visualizations(training_results, evaluation_results, data_dict, data_loader)
            
            # Step 5: Save results
            if save_results:
                self.save_experiment_results(training_results, evaluation_results, summary, data_dict)
            
            # Print final summary
            self.print_final_summary(summary, training_results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"\n⏱️  Total experiment duration: {duration}")
            
            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'summary': summary,
                'data_dict': data_dict,
                'experiment_dir': getattr(self, 'experiment_dir', None) if save_results else None
            }
            
        except Exception as e:
            print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
            raise


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Run quantile analysis experiment')
    parser.add_argument('--experiment', '-e', required=True, 
                       help='Name of experiment to run (from experiment_config.yaml)')
    parser.add_argument('--config_dir', '-c', default='../config',
                       help='Directory containing configuration files')
    parser.add_argument('--results_dir', '-r', default='../results',
                       help='Directory to save results')
    parser.add_argument('--no_save', action='store_true',
                       help='Run experiment without saving results')
    
    args = parser.parse_args()
    
    # Initialize and run experiment
    runner = ExperimentRunner(
        experiment_name=args.experiment,
        config_dir=args.config_dir,
        results_dir=args.results_dir
    )
    
    # Run experiment
    results = runner.run_full_experiment(save_results=not args.no_save)
    
    return results


if __name__ == "__main__":
    main() 