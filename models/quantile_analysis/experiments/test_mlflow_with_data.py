#!/usr/bin/env python3
"""
Quick MLflow Test with Real Data

Tests MLflow integration with your actual quantile analysis data.
This is a lightweight version to validate everything works.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import mlflow
import mlflow.lightgbm
import pandas as pd
from datetime import datetime
import time

# Import your existing modules
from data_loader import DataLoader
from model_trainer import QuantileModelTrainer
from evaluator import QuantileEvaluator
from utils import load_config
from tracking import ExperimentTracker

def quick_data_test():
    """Quick test with your actual data"""
    
    print("🚀 Quick MLflow Test with Your Data")
    print("=" * 50)
    
    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("quick_data_test")
    
    try:
        # Load your configurations
        config_dir = 'config'
        data_config = load_config(os.path.join(config_dir, 'data_config.yaml'))
        model_config = load_config(os.path.join(config_dir, 'model_config.yaml'))
        experiment_config = load_config(os.path.join(config_dir, 'experiment_config.yaml'))
        
        print("✅ Loaded configurations")
        
        # Load your data
        print("📂 Loading your actual data...")
        data_loader = DataLoader(data_config, experiment_config)
        
        # Use the nav_baseline experiment (or first available)
        available_experiments = list(experiment_config['experiments'].keys())
        experiment_name = available_experiments[0]  # Usually 'nav_baseline'
        
        print(f"Using experiment: {experiment_name}")
        
        data_dict = data_loader.prepare_experiment_data(experiment_name)
        
        print(f"✅ Data loaded:")
        print(f"  Total samples: {len(data_dict['df_full'])}")
        print(f"  Training samples: {len(data_dict['X_train'])}")
        print(f"  Test samples: {len(data_dict['X_test'])}")
        print(f"  Features: {len(data_dict['feature_names'])}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"quick_test_{datetime.now().strftime('%H%M%S')}") as run:
            print(f"\n📊 Started MLflow run: {run.info.run_id}")
            
            # Log basic data info
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("total_samples", len(data_dict['df_full']))
            mlflow.log_param("train_samples", len(data_dict['X_train']))
            mlflow.log_param("test_samples", len(data_dict['X_test']))
            mlflow.log_param("n_features", len(data_dict['feature_names']))
            mlflow.log_param("feature_names", str(data_dict['feature_names']))
            
            # Log data quality metrics
            df = data_dict['df_full']
            mlflow.log_metric("missing_values", df.isnull().sum().sum())
            mlflow.log_metric("missing_percentage", (df.isnull().sum().sum() / df.size) * 100)
            mlflow.log_metric("duplicates", df.duplicated().sum())
            
            # Quick model training (just 3 quantiles for speed)
            print("🤖 Training a quick model...")
            start_time = time.time()
            
            trainer = QuantileModelTrainer(model_config)
            exp_config = experiment_config['experiments'][experiment_name]
            
            # Use a small set of quantiles for quick testing
            quick_quantiles = 'sparse'  # [0.1, 0.5, 0.9]
            
            training_results = trainer.train_quantile_models(
                data_dict['X_train'][:1000],  # Use subset for speed
                data_dict['y_train'][:1000],  # Use subset for speed
                exp_config['model_type'], 
                data_dict['feature_names'],
                quantile_set=quick_quantiles
            )
            
            training_time = time.time() - start_time
            
            print(f"✅ Models trained for {len(training_results['quantiles'])} quantiles")
            print(f"  Training time: {training_time:.2f} seconds")
            
            # Log training info
            mlflow.log_param("model_type", training_results['model_type'])
            mlflow.log_param("quantiles_trained", str(training_results['quantiles']))
            mlflow.log_metric("training_time_seconds", training_time)
            mlflow.log_metric("samples_used_for_training", 1000)
            
            # Quick evaluation
            print("📊 Quick evaluation...")
            evaluator = QuantileEvaluator(data_config=data_config)
            
            target_name = exp_config['target']
            target_config = experiment_config['targets'][target_name]
            target_column = target_config['column']
            
            # Use subset for quick evaluation
            quick_data_dict = {
                'X_train': data_dict['X_train'][:1000],
                'y_train': data_dict['y_train'][:1000],
                'X_test': data_dict['X_test'][:200],
                'y_test': data_dict['y_test'][:200],
                'df_full': data_dict['df_full']
            }
            
            evaluation_results = evaluator.evaluate_models(
                training_results, quick_data_dict, target_column
            )
            
            # Log key metrics
            mlflow.log_metric("avg_pinball_loss", evaluation_results.get('average_pinball_loss', 0))
            mlflow.log_metric("avg_coverage", evaluation_results.get('average_coverage', 0))
            
            # Log feature importance if available
            if 'feature_importance' in training_results:
                importance_dict = training_results['feature_importance']
                # Average importance across quantiles for the median
                if 0.5 in importance_dict:
                    median_importance = importance_dict[0.5]
                    for feature, importance in median_importance.items():
                        mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            # Create simple summary artifact
            summary_text = f"""
Quick MLflow Test Summary
========================
Timestamp: {datetime.now().isoformat()}
Experiment: {experiment_name}
Data file: {data_config.get('file_path', 'unknown')}

Dataset Info:
- Total samples: {len(data_dict['df_full'])}
- Features: {len(data_dict['feature_names'])}
- Missing values: {df.isnull().sum().sum()}

Model Training:
- Model type: {training_results['model_type']}
- Quantiles: {training_results['quantiles']}
- Training samples used: 1000
- Training time: {training_time:.2f} seconds

Quick Evaluation:
- Test samples used: 200
- Average pinball loss: {evaluation_results.get('average_pinball_loss', 'N/A')}
- Average coverage: {evaluation_results.get('average_coverage', 'N/A')}

Features used:
{chr(10).join([f'- {f}' for f in data_dict['feature_names']])}
            """
            
            with open("quick_test_summary.txt", "w") as f:
                f.write(summary_text)
            mlflow.log_artifact("quick_test_summary.txt")
            os.remove("quick_test_summary.txt")
            
            print("📈 Results:")
            avg_pinball = evaluation_results.get('average_pinball_loss', 'N/A')
            avg_coverage = evaluation_results.get('average_coverage', 'N/A')
            
            if isinstance(avg_pinball, (int, float)):
                print(f"  Average pinball loss: {avg_pinball:.4f}")
            else:
                print(f"  Average pinball loss: {avg_pinball}")
                
            if isinstance(avg_coverage, (int, float)):
                print(f"  Average coverage: {avg_coverage:.4f}")
            else:
                print(f"  Average coverage: {avg_coverage}")
                
            print(f"  Training time: {training_time:.2f} seconds")
            
            print(f"\n✅ MLflow run completed: {run.info.run_id}")
            print("📊 View results at: http://localhost:5000")
            
            return {
                'run_id': run.info.run_id,
                'experiment_name': experiment_name,
                'training_time': training_time,
                'evaluation_results': evaluation_results
            }
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def test_experiment_tracker():
    """Test the ExperimentTracker integration"""
    
    print("\n🧪 Testing ExperimentTracker Integration")
    print("=" * 50)
    
    try:
        # Initialize tracker
        tracker = ExperimentTracker()
        
        if not tracker.is_enabled():
            print("⚠️  MLflow tracking not enabled")
            return
        
        print("✅ ExperimentTracker initialized")
        
        # Test basic functionality
        test_config = {
            'model_type': 'lightgbm',
            'target': 'nav_log',
            'test_mode': True
        }
        
        # Start experiment
        success = tracker.start_experiment("integration_test", test_config)
        print(f"Start experiment success: {success}")
        
        if success:
            # Create dummy data for testing
            import numpy as np
            dummy_data = pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'target': np.random.randn(100)
            })
            
            # Test data logging
            tracker.log_data_stage(dummy_data, "test_data", {"source": "integration_test"})
            print("✅ Data logging works")
            
            # Test summary logging
            test_summary = {
                'test_metric': 0.123,
                'test_param': 'success'
            }
            tracker.log_experiment_summary(test_summary)
            print("✅ Summary logging works")
            
            # End experiment
            tracker.end_experiment(success=True)
            print("✅ Experiment ended successfully")
            
            run_info = tracker.get_run_info()
            if run_info:
                print(f"✅ Run info retrieved: {run_info['run_id']}")
        
        print("✅ ExperimentTracker integration test completed")
        
    except Exception as e:
        print(f"❌ ExperimentTracker test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("🎯 Running MLflow Integration Tests")
    print("=" * 60)
    
    try:
        # Test 1: Quick data test
        result1 = quick_data_test()
        
        # Test 2: ExperimentTracker test
        test_experiment_tracker()
        
        print("\n🎉 All tests completed successfully!")
        print("📊 Check your MLflow UI at: http://localhost:5000")
        print("\nYou should see:")
        print("  1. A 'quick_data_test' experiment with real data analysis")
        print("  2. An 'integration_test' experiment with tracker test")
        
        return result1
        
    except Exception as e:
        print(f"\n❌ Tests failed: {str(e)}")
        return None

if __name__ == "__main__":
    main() 