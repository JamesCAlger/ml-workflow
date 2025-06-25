import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import os

class QuantileVisualizer:
    """Visualization tools for quantile analysis"""
    
    def __init__(self, output_path=None, style='seaborn-v0_8', data_config=None, transformation_manager=None):
        """Initialize visualizer with output settings, data configuration, and transformation manager"""
        self.output_path = output_path
        self.data_config = data_config
        self.transformation_manager = transformation_manager
        self.target_column = None  # Will be set during visualization
        
        # Get aggregation covariate column
        if data_config:
            aggregation_config = data_config['preprocessing'].get('aggregation', {})
            self.aggregation_col = aggregation_config.get('covariate_column', 'strategy')
        else:
            self.aggregation_col = 'strategy'  # Default fallback
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        if output_path:
            self.plots_dir = os.path.join(output_path, "plots")
            os.makedirs(self.plots_dir, exist_ok=True)
    
    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """Save plot to output directory"""
        if self.plots_dir:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Plot saved: {filepath}")
    
    def set_target_column(self, target_column):
        """Set the target column for transformation-aware visualization"""
        self.target_column = target_column
    
    def get_visualization_info(self, target_column=None):
        """Get information about how to handle visualization for the current target"""
        if target_column is None:
            target_column = self.target_column
        
        if not self.transformation_manager or not target_column:
            return {
                'actual_column': 'disbursement',  # Default fallback
                'needs_prediction_transform': False,
                'transform_complexity': 'none',
                'warning': None
            }
        
        return self.transformation_manager.get_visualization_column_info(target_column)
    
    def plot_disbursements_by_age(self, df_disbursements, save=True):
        """Plot average disbursement by age and strategy"""
        print("Creating disbursements by age plot...")
        
        # Group by strategy and age, calculate mean disbursements
        plot_data = df_disbursements.groupby([self.aggregation_col, 'age'])['disbursement'].mean().reset_index()
        
        # Get unique strategies
        strategies = plot_data[self.aggregation_col].unique()
        
        # Create subplots - one for each strategy
        fig, axes = plt.subplots(len(strategies), 1, figsize=(12, 6*len(strategies)))
        
        # If there's only one strategy, axes won't be an array
        if len(strategies) == 1:
            axes = [axes]
        
        # Plot each strategy in its own subplot
        for i, strategy in enumerate(strategies):
            strategy_data = plot_data[plot_data[self.aggregation_col] == strategy]
            axes[i].plot(strategy_data['age'], strategy_data['disbursement'], 
                         linewidth=2, alpha=0.8, marker='o', markersize=4)
            
            axes[i].set_xlabel('Age (Quarters since first investment)', fontsize=12)
            axes[i].set_ylabel('Average Disbursement', fontsize=12)
            axes[i].set_title(f'Average Disbursements by Investment Age - {strategy} {self.aggregation_col.title()}', fontsize=14)
            axes[i].grid(True, alpha=0.3)
            
            # Format y-axis to show values in millions
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        
        if save:
            self.save_plot(f"disbursements_by_age_{self.aggregation_col}.png")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance, quantiles, feature_names, save=True):
        """Plot feature importance across quantiles"""
        print("Creating feature importance plots...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, quantile in enumerate(quantiles):
            ax = axes[i]
            
            # Get feature importance for this quantile
            if isinstance(feature_importance[quantile], dict):
                importance = [feature_importance[quantile][f] for f in feature_names]
            else:
                importance = feature_importance[quantile]
            
            # Create bar plot
            bars = ax.bar(feature_names, importance, alpha=0.7)
            ax.set_title(f'Feature Importance - Q{int(quantile*100)}', fontsize=12)
            ax.set_ylabel('Importance (Gain)', fontsize=10)
            
            # Rotate x labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, importance):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance)*0.01,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplots
        for j in range(len(quantiles), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Feature Importance Across All Quantiles', fontsize=16, y=1.02)
        
        if save:
            self.save_plot("feature_importance_by_quantile.png")
        
        plt.show()
    
    def plot_residual_analysis(self, y_test, predictions_test, quantiles, save=True):
        """Create QQ plots and residual analysis"""
        print("Creating residual analysis plots...")
        
        # Use median quantile for residual analysis
        if 0.5 not in quantiles:
            print("Warning: Median quantile (0.5) not available for residual analysis")
            return
        
        median_predictions = predictions_test[0.5]
        residuals = y_test.values - median_predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # QQ plot
        stats.probplot(residuals, dist="norm", plot=axes[0,0])
        axes[0,0].set_title('Q-Q Plot: Residuals vs Normal Distribution', fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # Residuals vs Predicted
        axes[0,1].scatter(median_predictions, residuals, alpha=0.6, s=30)
        axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0,1].set_xlabel('Predicted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals vs Predicted Values', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1,0].hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Distribution of Residuals', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        
        # Overlay normal distribution for comparison
        mu, sigma = norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1,0].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                       label=f'Normal(μ={mu:.0f}, σ={sigma:.0f})')
        axes[1,0].legend()
        
        # Coverage calibration plot
        coverage_quantiles = []
        coverage_empirical = []
        
        for q in quantiles:
            if q <= 0.5:
                # For lower quantiles, check if actual values are below predicted
                coverage = np.mean(y_test.values <= predictions_test[q])
            else:
                # For upper quantiles, check if actual values are above predicted  
                coverage = np.mean(y_test.values >= predictions_test[q])
            
            coverage_quantiles.append(q)
            coverage_empirical.append(coverage)
        
        axes[1,1].plot(coverage_quantiles, coverage_quantiles, 'r--', label='Perfect Calibration', linewidth=2)
        axes[1,1].plot(coverage_quantiles, coverage_empirical, 'bo-', label='Empirical Coverage', markersize=6)
        axes[1,1].set_xlabel('Theoretical Quantile')
        axes[1,1].set_ylabel('Empirical Coverage')
        axes[1,1].set_title('Coverage Calibration Plot', fontsize=12)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            self.save_plot("residual_analysis.png")
        
        plt.show()
    
    def plot_prediction_comparison(self, df_full, predictions_full, save=True):
        """Create predicted vs actual comparison plots by age"""
        print("Creating prediction comparison plots...")
        
        if 0.5 not in predictions_full:
            print("Warning: Median quantile (0.5) not available for comparison")
            return
        
        # Get visualization information to handle transformations properly
        viz_info = self.get_visualization_info()
        actual_column = viz_info['actual_column']
        transform_complexity = viz_info['transform_complexity']
        
        # Print warning if transformation is complex
        if 'warning' in viz_info and viz_info['warning']:
            print(f"Visualization Warning: {viz_info['warning']}")
            if transform_complexity == 'complex_chain':
                print("Using transformed scale for both actual and predicted values")
                actual_column = self.target_column if self.target_column and self.target_column in df_full.columns else 'disbursement'
        
        # Check if the actual column exists
        if actual_column not in df_full.columns:
            print(f"Warning: Column '{actual_column}' not found in dataframe. Available columns: {list(df_full.columns)}")
            # Fallback to target column if it exists
            if self.target_column and self.target_column in df_full.columns:
                actual_column = self.target_column
                print(f"Using target column '{actual_column}' for visualization")
            else:
                print("Using 'disbursement' as fallback")
                actual_column = 'disbursement'
        
        # Calculate actual averages by age and strategy
        actual_by_age_strategy = df_full.groupby([self.aggregation_col, 'age'])[actual_column].mean().reset_index()
        
        # Add predictions to dataframe and calculate predicted averages
        df_with_preds = df_full.copy()
        df_with_preds['pred_q50'] = predictions_full[0.5]
        predicted_by_age_strategy = df_with_preds.groupby([self.aggregation_col, 'age'])['pred_q50'].mean().reset_index()
        
        # Get unique strategies for plotting
        unique_strategies = actual_by_age_strategy[self.aggregation_col].unique()
        
        # Create subplots for comparison
        fig, axes = plt.subplots(len(unique_strategies), 1, figsize=(14, 6*len(unique_strategies)))
        
        if len(unique_strategies) == 1:
            axes = [axes]
        
        # Determine labels and formatting based on actual column
        if actual_column == 'nav':
            y_label = 'Average NAV'
            title_prefix = 'NAV'
            value_formatter = lambda x, p: f'${x/1e6:.1f}M'
        elif actual_column == 'disbursement':
            y_label = 'Average Disbursement'
            title_prefix = 'Disbursements'
            value_formatter = lambda x, p: f'${x/1e6:.1f}M'
        elif 'log' in actual_column and 'diff' in actual_column:
            y_label = f'Average {actual_column.replace("_", " ").title()}'
            title_prefix = 'Log-Differenced Values'
            value_formatter = lambda x, p: f'{x:.3f}'
        elif 'log' in actual_column:
            y_label = f'Average {actual_column.replace("_", " ").title()}'
            title_prefix = 'Log-Transformed Values'
            value_formatter = lambda x, p: f'{x:.2f}'
        else:
            y_label = f'Average {actual_column.replace("_", " ").title()}'
            title_prefix = actual_column.replace("_", " ").title()
            value_formatter = lambda x, p: f'{x:.2f}'
        
        for i, strategy in enumerate(unique_strategies):
            # Get data for this strategy
            actual_strategy = actual_by_age_strategy[actual_by_age_strategy[self.aggregation_col] == strategy]
            predicted_strategy = predicted_by_age_strategy[predicted_by_age_strategy[self.aggregation_col] == strategy]
            
            # Plot actual vs predicted
            axes[i].plot(actual_strategy['age'], actual_strategy[actual_column], 
                         'o-', linewidth=2, markersize=6, label='Actual Average', alpha=0.8)
            axes[i].plot(predicted_strategy['age'], predicted_strategy['pred_q50'], 
                         's-', linewidth=2, markersize=6, label='Predicted Average (Q50)', alpha=0.8)
            
            axes[i].set_xlabel('Age (Quarters since first investment)', fontsize=12)
            axes[i].set_ylabel(y_label, fontsize=12)
            axes[i].set_title(f'Predicted vs Actual Average {title_prefix} by Age - {strategy} {self.aggregation_col.title()}', fontsize=14)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Format y-axis appropriately
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(value_formatter))
        
        plt.tight_layout()
        
        if save:
            self.save_plot("prediction_vs_actual_comparison.png")
        
        plt.show()
    
    def plot_prediction_intervals(self, df_full, predictions_full, quantiles, save=True):
        """Create prediction interval visualization"""
        print("Creating prediction intervals visualization...")
        
        # Check required quantiles
        required_quantiles = [0.1, 0.2, 0.5, 0.8, 0.9]
        available_quantiles = [q for q in required_quantiles if q in quantiles]
        
        if len(available_quantiles) < 3:
            print("Warning: Not enough quantiles available for interval visualization")
            return
        
        unique_strategies = df_full[self.aggregation_col].unique()
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        strategies_for_viz = unique_strategies[:8]  # Show up to 8 strategies
        
        for i, strategy in enumerate(strategies_for_viz):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get data for this strategy
            strategy_mask = df_full[self.aggregation_col] == strategy
            strategy_data = df_full[strategy_mask].copy()
            
            # Sort by age for smooth plotting
            strategy_data = strategy_data.sort_values('age')
            
            # Add predictions using boolean indexing to align with the full predictions
            if 0.5 in predictions_full:
                strategy_predictions_q50 = predictions_full[0.5][strategy_mask]
                # Sort the predictions in the same order as the sorted strategy data
                strategy_sort_order = strategy_data.index
                full_sort_order = df_full[strategy_mask].index
                
                # Create a mapping from original index to sorted position
                pred_dict = dict(zip(full_sort_order, strategy_predictions_q50))
                strategy_data['pred_q50'] = [pred_dict[idx] for idx in strategy_sort_order]
                
                ax.plot(strategy_data['age'], strategy_data['pred_q50'], 'b-', 
                       linewidth=2, label='Median (Q50)')
            
            # Plot prediction intervals if available
            if 0.1 in predictions_full and 0.9 in predictions_full:
                strategy_predictions_q10 = predictions_full[0.1][strategy_mask]
                strategy_predictions_q90 = predictions_full[0.9][strategy_mask]
                
                pred_dict_q10 = dict(zip(full_sort_order, strategy_predictions_q10))
                pred_dict_q90 = dict(zip(full_sort_order, strategy_predictions_q90))
                
                strategy_data['pred_q10'] = [pred_dict_q10[idx] for idx in strategy_sort_order]
                strategy_data['pred_q90'] = [pred_dict_q90[idx] for idx in strategy_sort_order]
                
                ax.fill_between(strategy_data['age'], strategy_data['pred_q10'], strategy_data['pred_q90'], 
                               alpha=0.2, color='blue', label='80% Interval (Q10-Q90)')
            
            if 0.2 in predictions_full and 0.8 in predictions_full:
                strategy_predictions_q20 = predictions_full[0.2][strategy_mask]
                strategy_predictions_q80 = predictions_full[0.8][strategy_mask]
                
                pred_dict_q20 = dict(zip(full_sort_order, strategy_predictions_q20))
                pred_dict_q80 = dict(zip(full_sort_order, strategy_predictions_q80))
                
                strategy_data['pred_q20'] = [pred_dict_q20[idx] for idx in strategy_sort_order]
                strategy_data['pred_q80'] = [pred_dict_q80[idx] for idx in strategy_sort_order]
                
                ax.fill_between(strategy_data['age'], strategy_data['pred_q20'], strategy_data['pred_q80'], 
                               alpha=0.3, color='blue', label='60% Interval (Q20-Q80)')
            
            # Get visualization info for appropriate actual column
            viz_info = self.get_visualization_info()
            actual_column = viz_info['actual_column']
            if actual_column not in strategy_data.columns:
                actual_column = 'disbursement'  # Fallback
            
            # Plot actual values
            ax.scatter(strategy_data['age'], strategy_data[actual_column], 
                      alpha=0.4, s=20, color='red', label='Actual')
            
            # Determine appropriate labels and formatting
            if actual_column == 'nav':
                y_label = 'NAV'
                value_formatter = lambda x, p: f'${x/1e6:.1f}M'
            elif actual_column == 'disbursement':
                y_label = 'Disbursement'
                value_formatter = lambda x, p: f'${x/1e6:.1f}M'
            elif 'log' in actual_column and 'diff' in actual_column:
                y_label = f'{actual_column.replace("_", " ").title()}'
                value_formatter = lambda x, p: f'{x:.3f}'
            elif 'log' in actual_column:
                y_label = f'{actual_column.replace("_", " ").title()}'
                value_formatter = lambda x, p: f'{x:.2f}'
            else:
                y_label = f'{actual_column.replace("_", " ").title()}'
                value_formatter = lambda x, p: f'{x:.2f}'
            
            ax.set_xlabel('Age (Quarters)')
            ax.set_ylabel(y_label)
            ax.set_title(f'{strategy} - Prediction Intervals', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Format y-axis appropriately
            ax.yaxis.set_major_formatter(plt.FuncFormatter(value_formatter))
        
        # Hide unused subplots
        for j in range(len(strategies_for_viz), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            self.save_plot(f"prediction_intervals_by_{self.aggregation_col}.png")
        
        plt.show()
    
    def plot_coverage_analysis(self, coverage_df, save=True):
        """Plot coverage analysis results"""
        print("Creating coverage analysis plot...")
        
        if coverage_df.empty:
            print("No coverage data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Coverage difference plot
        intervals = coverage_df['Interval']
        coverage_diff = coverage_df['Coverage_Difference']
        expected_coverage = coverage_df['Expected_Coverage']
        empirical_coverage = coverage_df['Empirical_Coverage']
        
        ax1.bar(intervals, coverage_diff, alpha=0.7, 
               color=['red' if x < 0 else 'green' for x in coverage_diff])
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax1.set_xlabel('Prediction Interval')
        ax1.set_ylabel('Coverage Difference (Empirical - Expected)')
        ax1.set_title('Coverage Calibration Analysis')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(coverage_diff):
            ax1.text(i, v + 0.005 if v >= 0 else v - 0.015, f'{v:.3f}', 
                    ha='center', va='bottom' if v >= 0 else 'top')
        
        # Expected vs empirical coverage
        x_pos = np.arange(len(intervals))
        width = 0.35
        
        ax2.bar(x_pos - width/2, expected_coverage, width, 
               label='Expected Coverage', alpha=0.7)
        ax2.bar(x_pos + width/2, empirical_coverage, width, 
               label='Empirical Coverage', alpha=0.7)
        
        ax2.set_xlabel('Prediction Interval')
        ax2.set_ylabel('Coverage Rate')
        ax2.set_title('Expected vs Empirical Coverage')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(intervals)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            self.save_plot("coverage_analysis.png")
        
        plt.show()
    
    def plot_covariate_performance(self, covariate_perf_df, save=True):
        """Plot covariate-specific performance metrics"""
        print(f"Creating {self.aggregation_col} performance plot...")
        
        if covariate_perf_df.empty:
            print(f"No {self.aggregation_col} performance data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get the column name for the covariate
        covariate_col_name = self.aggregation_col.title()
        
        # Mean pinball loss by covariate
        axes[0,0].bar(covariate_perf_df[covariate_col_name], covariate_perf_df['Mean_Pinball_Loss'], alpha=0.7)
        axes[0,0].set_xlabel(covariate_col_name)
        axes[0,0].set_ylabel('Mean Pinball Loss')
        axes[0,0].set_title(f'Mean Pinball Loss by {covariate_col_name}')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Coverage by covariate
        if 'Coverage_80pct' in covariate_perf_df.columns:
            coverage_data = covariate_perf_df.dropna(subset=['Coverage_80pct'])
            if not coverage_data.empty:
                axes[0,1].bar(coverage_data[covariate_col_name], coverage_data['Coverage_80pct'], alpha=0.7)
                axes[0,1].axhline(y=0.8, color='red', linestyle='--', label='Target 80%')
                axes[0,1].set_xlabel(covariate_col_name)
                axes[0,1].set_ylabel('80% Coverage Rate')
                axes[0,1].set_title(f'80% Prediction Interval Coverage by {covariate_col_name}')
                axes[0,1].tick_params(axis='x', rotation=45)
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
        
        # Sample sizes
        axes[1,0].bar(covariate_perf_df[covariate_col_name], covariate_perf_df['N_samples'], alpha=0.7)
        axes[1,0].set_xlabel(covariate_col_name)
        axes[1,0].set_ylabel('Number of Test Samples')
        axes[1,0].set_title(f'Test Sample Size by {covariate_col_name}')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Median actual vs predicted
        if 'Median_Actual' in covariate_perf_df.columns and 'Median_Predicted' in covariate_perf_df.columns:
            pred_data = covariate_perf_df.dropna(subset=['Median_Actual', 'Median_Predicted'])
            if not pred_data.empty:
                x_pos = np.arange(len(pred_data))
                width = 0.35
                
                axes[1,1].bar(x_pos - width/2, pred_data['Median_Actual'], width, 
                             label='Actual', alpha=0.7)
                axes[1,1].bar(x_pos + width/2, pred_data['Median_Predicted'], width, 
                             label='Predicted', alpha=0.7)
                
                axes[1,1].set_xlabel(covariate_col_name)
                axes[1,1].set_ylabel('Median Disbursement')
                axes[1,1].set_title(f'Median Actual vs Predicted by {covariate_col_name}')
                axes[1,1].set_xticks(x_pos)
                axes[1,1].set_xticklabels(pred_data[covariate_col_name], rotation=45)
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            self.save_plot(f"{self.aggregation_col}_performance_analysis.png")
        
        plt.show()
    
    def create_all_visualizations(self, df_full, evaluation_results, training_results, save=True, target_column=None):
        """Create all standard visualizations for the analysis"""
        print("Creating all visualizations...")
        
        # Set target column for transformation-aware visualization
        if target_column:
            self.set_target_column(target_column)
        
        predictions = evaluation_results['predictions']
        
        # 1. Disbursements by age (if full data available)
        if df_full is not None:
            self.plot_disbursements_by_age(df_full, save=save)
        
        # 2. Feature importance
        feature_importance = training_results['feature_importance']
        quantiles = training_results['quantiles']
        feature_names = training_results['feature_names']
        self.plot_feature_importance(feature_importance, quantiles, feature_names, save=save)
        
        # 3. Residual analysis
        if 'test' in predictions:
            self.plot_residual_analysis(
                evaluation_results.get('y_test'), predictions['test'], quantiles, save=save
            )
        
        # 4. Prediction comparison
        if df_full is not None and 'full' in predictions:
            self.plot_prediction_comparison(df_full, predictions['full'], save=save)
        
        # 5. Prediction intervals
        if df_full is not None and 'full' in predictions:
            self.plot_prediction_intervals(df_full, predictions['full'], quantiles, save=save)
        
        # 6. Coverage analysis
        coverage_df = evaluation_results.get('coverage_analysis')
        if coverage_df is not None and not coverage_df.empty:
            self.plot_coverage_analysis(coverage_df, save=save)
        
        # 7. Covariate performance
        covariate_perf_df = evaluation_results.get('covariate_performance')
        if covariate_perf_df is not None and not covariate_perf_df.empty:
            self.plot_covariate_performance(covariate_perf_df, save=save)
        
        print("All visualizations completed!") 