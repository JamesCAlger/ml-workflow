import pandas as pd
import numpy as np
from scipy import stats
from .base_transform import BaseTransform

class BoxCoxTransform(BaseTransform):
    """Apply Box-Cox transformation to normalize data and stabilize variance"""
    
    def __init__(self, lambda_param=None, optimize_lambda=True, handle_zeros='add_constant', **kwargs):
        """
        Initialize Box-Cox transformation
        
        Args:
            lambda_param: Box-Cox parameter. If None and optimize_lambda=True, will be estimated
            optimize_lambda: Whether to optimize lambda parameter automatically
            handle_zeros: How to handle zero/negative values:
                - 'add_constant': Add small constant to make all values positive
                - 'remove': Remove zero/negative values
        """
        super().__init__(**kwargs)
        self.lambda_param = lambda_param
        self.optimize_lambda = optimize_lambda
        self.handle_zeros = handle_zeros
        self.fitted_lambda = None
        self.shift_constant = 0
    
    def fit(self, df, column_name):
        """Fit the Box-Cox transformation (estimate lambda if needed)"""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        data = df[column_name].copy()
        
        # Handle zero/negative values
        if (data <= 0).any():
            zero_negative_count = (data <= 0).sum()
            print(f"Found {zero_negative_count} zero/negative values in '{column_name}'")
            
            if self.handle_zeros == 'add_constant':
                # Add constant to make all values positive
                self.shift_constant = abs(data.min()) + 1
                data = data + self.shift_constant
                print(f"Added constant {self.shift_constant} to make all values positive")
                
            elif self.handle_zeros == 'remove':
                data = data[data > 0]
                print(f"Removed {zero_negative_count} zero/negative values")
                
            else:
                raise ValueError(f"Unknown handle_zeros method: {self.handle_zeros}")
        
        # Estimate lambda parameter if needed
        if self.optimize_lambda and self.lambda_param is None:
            # Use scipy's boxcox to find optimal lambda
            _, self.fitted_lambda = stats.boxcox(data)
            print(f"Optimized Box-Cox lambda: {self.fitted_lambda:.4f}")
        else:
            self.fitted_lambda = self.lambda_param or 0.0
            print(f"Using specified Box-Cox lambda: {self.fitted_lambda}")
        
        self.fitted = True
        return self
    
    def transform(self, df, column_name):
        """Apply Box-Cox transformation"""
        if not self.fitted:
            raise ValueError("Transformation must be fitted before transform")
        
        df = df.copy()
        data = df[column_name]
        
        # Apply the same shift as during fitting
        if self.shift_constant > 0:
            data = data + self.shift_constant
        
        # Apply Box-Cox transformation
        if self.fitted_lambda == 0:
            # When lambda = 0, Box-Cox becomes log transformation
            transformed_data = np.log(data)
        else:
            # Standard Box-Cox transformation: (x^lambda - 1) / lambda
            transformed_data = (np.power(data, self.fitted_lambda) - 1) / self.fitted_lambda
        
        # Add new column with transformed values (preserve original)
        new_column_name = f"{column_name}_boxcox"
        df[new_column_name] = transformed_data
        
        print(f"Box-Cox transformation applied: '{column_name}' -> '{new_column_name}'")
        print(f"Lambda parameter: {self.fitted_lambda:.4f}")
        print(f"Original range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"Transformed range: [{transformed_data.min():.2f}, {transformed_data.max():.2f}]")
        print(f"Original column '{column_name}' preserved alongside '{new_column_name}'")
        
        return df
    
    def inverse_transform(self, values, original_column_name=None, group_ids=None):
        """Reverse the Box-Cox transformation to get back to original scale"""
        if not self.fitted:
            raise ValueError("Transformation must be fitted before inverse transform")
        
        # Reverse Box-Cox transformation
        if self.fitted_lambda == 0:
            # When lambda = 0, inverse is exp(x)
            original_values = np.exp(values)
        else:
            # Inverse Box-Cox: (lambda * x + 1)^(1/lambda)
            original_values = np.power(self.fitted_lambda * values + 1, 1 / self.fitted_lambda)
        
        # Remove the shift constant if it was applied
        if self.shift_constant > 0:
            original_values = original_values - self.shift_constant
        
        return original_values
    
    def get_metadata(self):
        """Get Box-Cox transformation metadata"""
        return {
            'complexity': 'simple',
            'visualization_scale': 'normalized',
            'requires_groups': False,
            'output_interpretation': 'normalized',
            'suggested_formatter': 'decimal'
        } 