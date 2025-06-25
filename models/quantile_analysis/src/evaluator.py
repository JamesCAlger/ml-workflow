import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import pinball_loss, empirical_coverage, prediction_interval_score

class QuantileEvaluator:
    """Comprehensive evaluation for quantile models"""
    
    def __init__(self, data_config=None, transformation_manager=None):
        """Initialize evaluator with data configuration and transformation manager"""
        self.evaluation_results = {}
        self.data_config = data_config
        self.transformation_manager = transformation_manager
        
        # Get aggregation covariate column
        if data_config:
            aggregation_config = data_config['preprocessing'].get('aggregation', {})
            self.aggregation_col = aggregation_config.get('covariate_column', 'strategy')
        else:
            self.aggregation_col = 'strategy'  # Default fallback
    
    def calculate_pinball_losses(self, y_true, predictions_train, predictions_test, y_train, quantiles):
        """Calculate pinball losses for all quantiles"""
        print("Calculating pinball losses...")
        
        pinball_losses_train = {}
        pinball_losses_test = {}
        
        for quantile in quantiles:
            # Calculate pinball loss for train set
            pinball_losses_train[quantile] = pinball_loss(
                y_train.values, predictions_train[quantile], quantile
            )
            
            # Calculate pinball loss for test set
            pinball_losses_test[quantile] = pinball_loss(
                y_true.values, predictions_test[quantile], quantile
            )
        
        # Create summary DataFrame
        loss_summary = pd.DataFrame({
            'Quantile': quantiles,
            'Train_Pinball_Loss': [pinball_losses_train[q] for q in quantiles],
            'Test_Pinball_Loss': [pinball_losses_test[q] for q in quantiles]
        })
        
        print("Pinball Loss Summary:")
        print(loss_summary.round(2))
        
        return {
            'loss_summary': loss_summary,
            'train_losses': pinball_losses_train,
            'test_losses': pinball_losses_test
        }
    
    def calculate_coverage_analysis(self, y_true, predictions_test, quantiles):
        """Calculate empirical coverage and prediction interval scores"""
        print("Calculating empirical coverage and prediction interval scores...")
        
        # Define prediction intervals
        intervals = [
            (0.1, 0.9, 0.2),  # 80% interval
            (0.2, 0.8, 0.4),  # 60% interval  
            (0.3, 0.7, 0.6),  # 40% interval
            (0.4, 0.6, 0.8),  # 20% interval
        ]
        
        coverage_results = []
        
        for lower_q, upper_q, alpha in intervals:
            # Skip if quantiles not available
            if lower_q not in quantiles or upper_q not in quantiles:
                continue
                
            # Get predictions for test set
            lower_pred = predictions_test[lower_q]
            upper_pred = predictions_test[upper_q]
            
            # Calculate empirical coverage
            coverage = empirical_coverage(y_true.values, lower_pred, upper_pred)
            
            # Calculate prediction interval score
            pis = prediction_interval_score(y_true.values, lower_pred, upper_pred, alpha)
            
            coverage_results.append({
                'Interval': f'{int((1-alpha)*100)}%',
                'Lower_Quantile': lower_q,
                'Upper_Quantile': upper_q,
                'Expected_Coverage': 1-alpha,
                'Empirical_Coverage': coverage,
                'Coverage_Difference': coverage - (1-alpha),
                'Prediction_Interval_Score': pis
            })
        
        coverage_df = pd.DataFrame(coverage_results)
        print("Coverage Analysis Results:")
        print(coverage_df.round(4))
        
        return coverage_df
    
    def analyze_performance_by_covariate(self, y_true, predictions_test, quantiles, df_test):
        """Analyze performance metrics by configurable covariate"""
        print(f"Analyzing performance by {self.aggregation_col}...")
        
        if df_test is None:
            print(f"Warning: No {self.aggregation_col} information available for analysis")
            return pd.DataFrame()
        
        # Get covariate information
        test_covariates = df_test[self.aggregation_col].values
        
        covariate_performance = []
        
        for covariate_value in df_test[self.aggregation_col].unique():
            covariate_mask = (test_covariates == covariate_value)
            covariate_y_true = y_true.values[covariate_mask]
            
            if len(covariate_y_true) == 0:
                continue
            
            # Calculate pinball losses for this covariate value
            covariate_losses = {}
            for quantile in quantiles:
                covariate_pred = predictions_test[quantile][covariate_mask]
                covariate_losses[quantile] = pinball_loss(covariate_y_true, covariate_pred, quantile)
            
            # Calculate coverage for 80% interval (if available)
            covariate_coverage = None
            if 0.1 in quantiles and 0.9 in quantiles:
                lower_pred = predictions_test[0.1][covariate_mask]
                upper_pred = predictions_test[0.9][covariate_mask]
                covariate_coverage = empirical_coverage(covariate_y_true, lower_pred, upper_pred)
            
            # Calculate median predictions vs actual
            median_actual = np.median(covariate_y_true)
            median_predicted = None
            if 0.5 in quantiles:
                median_predicted = np.median(predictions_test[0.5][covariate_mask])
            
            covariate_performance.append({
                self.aggregation_col.title(): covariate_value,
                'N_samples': len(covariate_y_true),
                'Mean_Pinball_Loss': np.mean(list(covariate_losses.values())),
                'Coverage_80pct': covariate_coverage,
                'Median_Actual': median_actual,
                'Median_Predicted': median_predicted
            })
        
        covariate_perf_df = pd.DataFrame(covariate_performance)
        print(f"Performance by {self.aggregation_col.title()}:")
        print(covariate_perf_df.round(4))
        
        return covariate_perf_df
    
    def calculate_correlation_analysis(self, actual_data, predicted_data, df_full, feature_names):
        """Calculate correlation between actual and predicted by covariate and age"""
        print(f"Calculating correlation analysis by {self.aggregation_col} and age...")
        
        if df_full is None:
            print("Warning: No full dataset available for correlation analysis")
            return pd.DataFrame()
        
        # Calculate actual averages by age and covariate
        actual_by_age_covariate = df_full.groupby([self.aggregation_col, 'age'])['disbursement'].mean().reset_index()
        
        # Create predictions for full dataset and calculate predicted averages
        if 0.5 not in predicted_data:
            print("Warning: Median quantile (0.5) not available for correlation analysis")
            return pd.DataFrame()
        
        # Add predictions to full dataset
        df_with_preds = df_full.copy()
        df_with_preds['pred_q50'] = predicted_data[0.5]
        
        predicted_by_age_covariate = df_with_preds.groupby([self.aggregation_col, 'age'])['pred_q50'].mean().reset_index()
        
        # Calculate correlations by covariate
        correlation_results = []
        unique_covariates = actual_by_age_covariate[self.aggregation_col].unique()
        
        for covariate_value in unique_covariates:
            actual_covariate = actual_by_age_covariate[actual_by_age_covariate[self.aggregation_col] == covariate_value]
            predicted_covariate = predicted_by_age_covariate[predicted_by_age_covariate[self.aggregation_col] == covariate_value]
            
            # Merge on age to ensure alignment
            merged = pd.merge(actual_covariate, predicted_covariate, on=[self.aggregation_col, 'age'], how='inner')
            
            if len(merged) > 1:
                correlation = np.corrcoef(merged['disbursement'], merged['pred_q50'])[0,1]
                mae = mean_absolute_error(merged['disbursement'], merged['pred_q50'])
                rmse = np.sqrt(mean_squared_error(merged['disbursement'], merged['pred_q50']))
                
                correlation_results.append({
                    self.aggregation_col.title(): covariate_value,
                    'Correlation': correlation,
                    'MAE': mae,
                    'RMSE': rmse,
                    'N_age_points': len(merged)
                })
        
        corr_df = pd.DataFrame(correlation_results)
        if not corr_df.empty:
            print(f"Predicted vs Actual Correlation Analysis by {self.aggregation_col.title()} (by Age):")
            print(corr_df.round(4))
        
        return corr_df
    
    def calculate_calibration_metrics(self, y_true, predictions_test, quantiles):
        """Calculate coverage calibration metrics"""
        coverage_quantiles = []
        coverage_empirical = []
        
        for q in quantiles:
            if q <= 0.5:
                # For lower quantiles, check if actual values are below predicted
                coverage = np.mean(y_true.values <= predictions_test[q])
            else:
                # For upper quantiles, check if actual values are above predicted  
                coverage = np.mean(y_true.values >= predictions_test[q])
            
            coverage_quantiles.append(q)
            coverage_empirical.append(coverage)
        
        calibration_df = pd.DataFrame({
            'Theoretical_Quantile': coverage_quantiles,
            'Empirical_Coverage': coverage_empirical,
            'Calibration_Error': np.array(coverage_empirical) - np.array(coverage_quantiles)
        })
        
        return calibration_df
    
    def evaluate_models(self, training_results, data_dict, target_column=None):
        """Comprehensive model evaluation with automatic back-transformation"""
        print("Starting comprehensive model evaluation...")
        
        # Extract data and predictions
        models = training_results['models']
        predictions_train = training_results['predictions_train']
        quantiles = training_results['quantiles']
        feature_names = training_results['feature_names']
        
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        y_train = data_dict['y_train']
        df_test = data_dict.get('df_test')
        df_full = data_dict.get('df_full')
        
        # Store original (transformed) predictions for log space evaluation
        predictions_train_log = predictions_train.copy()
        y_test_log = y_test.copy()
        y_train_log = y_train.copy()
        
        # Generate test predictions (in log space)
        predictions_test_log = {}
        for quantile, model in models.items():
            predictions_test_log[quantile] = model.predict(X_test)
        
        # Generate full dataset predictions for correlation analysis (in log space)
        full_predictions_log = {}
        if df_full is not None:
            X_full = df_full[feature_names]
            for quantile, model in models.items():
                full_predictions_log[quantile] = model.predict(X_full)
        
        # Automatic back-transformation if TransformationManager is available
        if self.transformation_manager and target_column:
            print("\n" + "="*60)
            print("APPLYING AUTOMATIC BACK-TRANSFORMATION")
            print("="*60)
            
            # Extract group information for log-difference transformations
            train_groups = None
            test_groups = None
            full_groups = None
            
            if 'group_info' in data_dict:
                group_info = data_dict['group_info']
                train_groups = group_info.get('train_groups')
                test_groups = group_info.get('test_groups')
                full_groups = group_info.get('full_groups')
                print(f"Found group information for back-transformation")
                print(f"Group column: {group_info.get('groupby_column', 'unknown')}")
            
            # Transform predictions back to original space
            predictions_train = self.transformation_manager.inverse_transform_predictions(
                predictions_train_log, target_column, group_ids=train_groups)
            predictions_test = self.transformation_manager.inverse_transform_predictions(
                predictions_test_log, target_column, group_ids=test_groups)
            full_predictions = self.transformation_manager.inverse_transform_predictions(
                full_predictions_log, target_column, group_ids=full_groups)
            
            # Transform target values back to original space for evaluation
            # We need the original target column name
            target_info = self.transformation_manager.get_target_info(target_column)
            original_column = target_info['original_column']
            
            # Use original disbursement column from dataframes if available
            if df_test is not None and original_column in df_test.columns:
                y_test = df_test[original_column]
                print(f"Using original target column '{original_column}' from test data")
            else:
                print(f"Warning: Original column '{original_column}' not found, using transformed values")
            
            if df_full is not None and original_column in df_full.columns:
                print(f"Original column '{original_column}' available in full dataset")
            
            print("✅ Back-transformation completed")
            print("="*60)
        else:
            # No back-transformation - use log space predictions
            predictions_test = predictions_test_log
            full_predictions = full_predictions_log
            print("No back-transformation applied - evaluating in transformed space")
        
        # Calculate all evaluation metrics
        evaluation_results = {}
        
        # 1. Pinball losses
        evaluation_results['pinball_analysis'] = self.calculate_pinball_losses(
            y_test, predictions_train, predictions_test, y_train, quantiles
        )
        
        # 2. Coverage analysis
        evaluation_results['coverage_analysis'] = self.calculate_coverage_analysis(
            y_test, predictions_test, quantiles
        )
        
        # 3. Covariate-specific performance
        evaluation_results['covariate_performance'] = self.analyze_performance_by_covariate(
            y_test, predictions_test, quantiles, df_test
        )
        
        # 4. Correlation analysis
        evaluation_results['correlation_analysis'] = self.calculate_correlation_analysis(
            df_full, full_predictions, df_full, feature_names
        )
        
        # 5. Calibration metrics
        evaluation_results['calibration_metrics'] = self.calculate_calibration_metrics(
            y_test, predictions_test, quantiles
        )
        
        # Store predictions for visualization
        evaluation_results['predictions'] = {
            'train': predictions_train,
            'test': predictions_test,
            'full': full_predictions
        }
        
        # Store test data for visualization
        evaluation_results['y_test'] = y_test
        
        print("Model evaluation completed!")
        
        return evaluation_results
    
    def generate_evaluation_summary(self, evaluation_results, training_results):
        """Generate a comprehensive evaluation summary"""
        quantiles = training_results['quantiles']
        model_type = training_results['model_type']
        feature_names = training_results['feature_names']
        
        # Extract key metrics
        pinball_analysis = evaluation_results['pinball_analysis']
        coverage_analysis = evaluation_results['coverage_analysis']
        covariate_performance = evaluation_results['covariate_performance']
        
        avg_test_pinball_loss = pinball_analysis['loss_summary']['Test_Pinball_Loss'].mean()
        
        # Find best coverage interval
        best_coverage_idx = coverage_analysis['Coverage_Difference'].abs().idxmin()
        best_coverage_interval = coverage_analysis.loc[best_coverage_idx, 'Interval']
        
        # Count covariate categories analyzed
        n_covariates = len(covariate_performance) if not covariate_performance.empty else 0
        
        summary = {
            'model_overview': {
                'n_quantiles': len(quantiles),
                'model_type': model_type,
                'features': feature_names,
                'n_covariates': n_covariates,
                'aggregation_covariate': self.aggregation_col
            },
            'performance_metrics': {
                'avg_test_pinball_loss': avg_test_pinball_loss,
                'best_coverage_interval': best_coverage_interval
            },
            'coverage_summary': coverage_analysis.to_dict('records') if not coverage_analysis.empty else [],
            'covariate_summary': covariate_performance.to_dict('records') if not covariate_performance.empty else []
        }
        
        return summary 