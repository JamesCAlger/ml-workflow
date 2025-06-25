#!/usr/bin/env python3
"""
Experiment comparison tool for quantile analysis

Usage:
    python compare_experiments.py --experiments baseline extended_features log_target
    python compare_experiments.py --results_dir ../results --output comparison_report
"""

import argparse
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, save_results

class ExperimentComparer:
    """Tool for comparing multiple experiments"""
    
    def __init__(self, results_dir='../results'):
        """Initialize experiment comparer"""
        self.results_dir = results_dir
        self.experiment_data = {}
        
    def load_experiment_results(self, experiment_names):
        """Load results from multiple experiments"""
        print(f"Loading results for {len(experiment_names)} experiments...")
        
        for exp_name in experiment_names:
            # Find the most recent experiment directory for this name
            exp_dir = self.find_latest_experiment_dir(exp_name)
            
            if exp_dir is None:
                print(f"Warning: No results found for experiment '{exp_name}'")
                continue
            
            # Load experiment data
            exp_data = self.load_single_experiment(exp_dir, exp_name)
            if exp_data:
                self.experiment_data[exp_name] = exp_data
                print(f"Loaded results for '{exp_name}' from {exp_dir}")
        
        print(f"Successfully loaded {len(self.experiment_data)} experiments")
        return self.experiment_data
    
    def find_latest_experiment_dir(self, experiment_name):
        """Find the most recent experiment directory for a given experiment name"""
        if not os.path.exists(self.results_dir):
            return None
        
        # Look for directories that start with experiment_name
        matching_dirs = []
        for dirname in os.listdir(self.results_dir):
            if dirname.startswith(f"{experiment_name}_"):
                full_path = os.path.join(self.results_dir, dirname)
                if os.path.isdir(full_path):
                    matching_dirs.append((dirname, full_path))
        
        if not matching_dirs:
            return None
        
        # Sort by directory name (which includes timestamp) and return the latest
        matching_dirs.sort(reverse=True)
        return matching_dirs[0][1]
    
    def load_single_experiment(self, exp_dir, exp_name):
        """Load data from a single experiment directory"""
        try:
            exp_data = {
                'name': exp_name,
                'directory': exp_dir
            }
            
            # Load summary
            summary_path = os.path.join(exp_dir, 'experiment_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    exp_data['summary'] = json.load(f)
            
            # Load evaluation results
            eval_path = os.path.join(exp_dir, 'evaluation_results.json')
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    exp_data['evaluation'] = json.load(f)
            
            # Load training metadata
            metadata_path = os.path.join(exp_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    exp_data['metadata'] = json.load(f)
            
            # Load configuration
            config_path = os.path.join(exp_dir, 'full_config.yaml')
            if os.path.exists(config_path):
                exp_data['config'] = load_config(config_path)
            
            return exp_data
            
        except Exception as e:
            print(f"Error loading experiment from {exp_dir}: {e}")
            return None
    
    def create_comparison_table(self):
        """Create comparison table of key metrics"""
        print("Creating comparison table...")
        
        comparison_data = []
        
        for exp_name, exp_data in self.experiment_data.items():
            row = {'Experiment': exp_name}
            
            # Add metadata
            if 'metadata' in exp_data:
                metadata = exp_data['metadata']
                row['Model_Type'] = metadata.get('model_type', 'Unknown')
                row['N_Quantiles'] = len(metadata.get('quantiles', []))
                row['Features'] = ', '.join(metadata.get('feature_names', []))
            
            # Add performance metrics
            if 'summary' in exp_data:
                summary = exp_data['summary']
                performance = summary.get('performance_metrics', {})
                row['Avg_Pinball_Loss'] = performance.get('avg_test_pinball_loss', None)
                row['Best_Coverage_Interval'] = performance.get('best_coverage_interval', None)
                
                model_overview = summary.get('model_overview', {})
                row['N_Strategies'] = model_overview.get('n_strategies', None)
            
            # Add coverage metrics
            if 'evaluation' in exp_data:
                coverage_analysis = exp_data['evaluation'].get('coverage_analysis', {})
                if isinstance(coverage_analysis, list) and coverage_analysis:
                    # Find 80% interval coverage
                    for coverage in coverage_analysis:
                        if coverage.get('Interval') == '80%':
                            row['Coverage_80pct'] = coverage.get('Empirical_Coverage', None)
                            row['Coverage_Diff_80pct'] = coverage.get('Coverage_Difference', None)
                            break
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("Comparison Table:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def create_performance_plots(self, output_dir=None):
        """Create performance comparison plots"""
        print("Creating performance comparison plots...")
        
        if len(self.experiment_data) < 2:
            print("Need at least 2 experiments for comparison plots")
            return
        
        # Prepare data for plotting
        plot_data = []
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'evaluation' in exp_data:
                eval_data = exp_data['evaluation']
                
                # Pinball losses
                if 'pinball_analysis' in eval_data:
                    pinball_data = eval_data['pinball_analysis']
                    if 'loss_summary' in pinball_data and isinstance(pinball_data['loss_summary'], list):
                        for loss_row in pinball_data['loss_summary']:
                            plot_data.append({
                                'Experiment': exp_name,
                                'Quantile': loss_row['Quantile'],
                                'Train_Loss': loss_row['Train_Pinball_Loss'],
                                'Test_Loss': loss_row['Test_Pinball_Loss']
                            })
        
        if not plot_data:
            print("No pinball loss data available for plotting")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Test pinball losses by quantile
        for exp_name in plot_df['Experiment'].unique():
            exp_subset = plot_df[plot_df['Experiment'] == exp_name]
            axes[0].plot(exp_subset['Quantile'], exp_subset['Test_Loss'], 
                        'o-', label=exp_name, linewidth=2, markersize=6)
        
        axes[0].set_xlabel('Quantile')
        axes[0].set_ylabel('Test Pinball Loss')
        axes[0].set_title('Test Pinball Loss by Quantile')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Average test loss comparison
        avg_losses = plot_df.groupby('Experiment')['Test_Loss'].mean().sort_values()
        axes[1].bar(range(len(avg_losses)), avg_losses.values, alpha=0.7)
        axes[1].set_xticks(range(len(avg_losses)))
        axes[1].set_xticklabels(avg_losses.index, rotation=45)
        axes[1].set_ylabel('Average Test Pinball Loss')
        axes[1].set_title('Average Test Pinball Loss by Experiment')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(avg_losses.values):
            axes[1].text(i, v + max(avg_losses.values)*0.01, f'{v:.0f}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = os.path.join(output_dir, "experiment_comparison_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved: {plot_path}")
        
        plt.show()
    
    def create_coverage_comparison(self, output_dir=None):
        """Create coverage comparison visualization"""
        print("Creating coverage comparison...")
        
        coverage_data = []
        
        for exp_name, exp_data in self.experiment_data.items():
            if 'evaluation' in exp_data:
                coverage_analysis = exp_data['evaluation'].get('coverage_analysis', {})
                if isinstance(coverage_analysis, list):
                    for coverage in coverage_analysis:
                        coverage_data.append({
                            'Experiment': exp_name,
                            'Interval': coverage.get('Interval', ''),
                            'Expected_Coverage': coverage.get('Expected_Coverage', 0),
                            'Empirical_Coverage': coverage.get('Empirical_Coverage', 0),
                            'Coverage_Difference': coverage.get('Coverage_Difference', 0)
                        })
        
        if not coverage_data:
            print("No coverage data available for comparison")
            return
        
        coverage_df = pd.DataFrame(coverage_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Coverage differences
        intervals = coverage_df['Interval'].unique()
        x_pos = range(len(intervals))
        width = 0.8 / len(self.experiment_data)
        
        for i, exp_name in enumerate(self.experiment_data.keys()):
            exp_coverage = coverage_df[coverage_df['Experiment'] == exp_name]
            if not exp_coverage.empty:
                positions = [x + width * i for x in x_pos]
                axes[0].bar(positions, exp_coverage['Coverage_Difference'], 
                           width, label=exp_name, alpha=0.7)
        
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.8)
        axes[0].set_xlabel('Prediction Interval')
        axes[0].set_ylabel('Coverage Difference (Empirical - Expected)')
        axes[0].set_title('Coverage Calibration Comparison')
        axes[0].set_xticks([x + width * (len(self.experiment_data) - 1) / 2 for x in x_pos])
        axes[0].set_xticklabels(intervals)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Empirical vs Expected Coverage
        for exp_name in self.experiment_data.keys():
            exp_coverage = coverage_df[coverage_df['Experiment'] == exp_name]
            if not exp_coverage.empty:
                axes[1].scatter(exp_coverage['Expected_Coverage'], 
                               exp_coverage['Empirical_Coverage'], 
                               label=exp_name, s=100, alpha=0.7)
        
        # Add perfect calibration line
        max_coverage = coverage_df['Expected_Coverage'].max()
        axes[1].plot([0, max_coverage], [0, max_coverage], 'r--', 
                    label='Perfect Calibration', linewidth=2)
        
        axes[1].set_xlabel('Expected Coverage')
        axes[1].set_ylabel('Empirical Coverage')
        axes[1].set_title('Coverage Calibration: Expected vs Empirical')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = os.path.join(output_dir, "coverage_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Coverage comparison saved: {plot_path}")
        
        plt.show()
    
    def generate_comparison_report(self, output_dir=None, output_name="comparison_report"):
        """Generate comprehensive comparison report"""
        print("Generating comparison report...")
        
        report = {
            'comparison_metadata': {
                'generated_at': datetime.now().isoformat(),
                'experiments_compared': list(self.experiment_data.keys()),
                'n_experiments': len(self.experiment_data)
            },
            'experiments': {}
        }
        
        # Add experiment details
        for exp_name, exp_data in self.experiment_data.items():
            exp_summary = {
                'name': exp_name,
                'directory': exp_data.get('directory', ''),
            }
            
            if 'metadata' in exp_data:
                exp_summary['metadata'] = exp_data['metadata']
            
            if 'summary' in exp_data:
                exp_summary['performance'] = exp_data['summary']
            
            report['experiments'][exp_name] = exp_summary
        
        # Create comparison table
        comparison_df = self.create_comparison_table()
        report['comparison_table'] = comparison_df.to_dict('records')
        
        # Save report
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON report
            json_path = os.path.join(output_dir, f"{output_name}.json")
            save_results(report, output_dir, f"{output_name}.json")
            
            # Save comparison table as CSV
            csv_path = os.path.join(output_dir, f"{output_name}_table.csv")
            comparison_df.to_csv(csv_path, index=False)
            
            print(f"Comparison report saved:")
            print(f"  JSON: {json_path}")
            print(f"  CSV:  {csv_path}")
        
        return report
    
    def run_full_comparison(self, experiment_names, output_dir=None, output_name="comparison_report"):
        """Run complete experiment comparison"""
        print("🔍 Starting Experiment Comparison")
        print(f"Experiments to compare: {experiment_names}")
        print("=" * 60)
        
        # Load experiment results
        self.load_experiment_results(experiment_names)
        
        if len(self.experiment_data) < 2:
            print("Error: Need at least 2 valid experiments for comparison")
            return None
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {output_dir}")
        
        # Generate comparison report
        report = self.generate_comparison_report(output_dir, output_name)
        
        # Create visualizations
        self.create_performance_plots(output_dir)
        self.create_coverage_comparison(output_dir)
        
        print("\n✅ COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return report


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Compare quantile analysis experiments')
    parser.add_argument('--experiments', '-e', nargs='+', required=True,
                       help='Names of experiments to compare')
    parser.add_argument('--results_dir', '-r', default='../results',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', '-o', 
                       help='Directory to save comparison results')
    parser.add_argument('--output_name', '-n', default='comparison_report',
                       help='Name for output files')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.results_dir, f"comparison_{timestamp}")
    
    # Initialize and run comparison
    comparer = ExperimentComparer(results_dir=args.results_dir)
    
    # Run comparison
    report = comparer.run_full_comparison(
        experiment_names=args.experiments,
        output_dir=args.output_dir,
        output_name=args.output_name
    )
    
    return report


if __name__ == "__main__":
    main() 