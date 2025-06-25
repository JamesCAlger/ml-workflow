import pandas as pd
import numpy as np
from .base_transform import BaseTransform

class LogTransform(BaseTransform):
    """Apply log transformation to a column"""
    
    def __init__(self, add_constant=1, handle_zeros='add_constant', **kwargs):
        """
        Initialize log transformation
        
        Args:
            add_constant: Constant to add before log transformation (default: 1)
            handle_zeros: How to handle zero/negative values:
                - 'add_constant': Add constant to all values
                - 'replace': Replace zeros/negatives with small constant
                - 'remove': Remove zero/negative values
        """
        super().__init__(**kwargs)
        self.add_constant = add_constant
        self.handle_zeros = handle_zeros
    
    def fit(self, df, column_name):
        """Fit the transformation (learn parameters if needed)"""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        # Check for zero/negative values
        data = df[column_name]
        zero_negative_count = (data <= 0).sum()
        
        if zero_negative_count > 0:
            print(f"Found {zero_negative_count} zero/negative values in '{column_name}'")
            
            if self.handle_zeros == 'replace':
                # Use minimum positive value / 2 as replacement
                min_positive = data[data > 0].min()
                self.replacement_value = min_positive / 2
                print(f"Will replace zero/negative values with {self.replacement_value}")
        
        self.fitted = True
        return self
    
    def transform(self, df, column_name):
        """Apply log transformation"""
        if not self.fitted:
            raise ValueError("Transformation must be fitted before transform")
        
        df = df.copy()
        data = df[column_name]
        
        # Handle zero/negative values
        if self.handle_zeros == 'add_constant':
            transformed_data = np.log(data + self.add_constant)
            print(f"Applied log({column_name} + {self.add_constant})")
            
        elif self.handle_zeros == 'replace':
            data_cleaned = data.copy()
            data_cleaned[data_cleaned <= 0] = self.replacement_value
            transformed_data = np.log(data_cleaned)
            print(f"Replaced {(data <= 0).sum()} zero/negative values and applied log")
            
        elif self.handle_zeros == 'remove':
            # Filter out zero/negative values
            mask = data > 0
            df = df[mask]
            transformed_data = np.log(df[column_name])
            print(f"Removed {(~mask).sum()} zero/negative values and applied log")
            
        else:
            raise ValueError(f"Unknown handle_zeros method: {self.handle_zeros}")
        
        # Replace original column with transformed values
        new_column_name = f"{column_name}_log"
        df[new_column_name] = transformed_data
        
        print(f"Log transformation applied: '{column_name}' -> '{new_column_name}'")
        print(f"Original range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"Transformed range: [{transformed_data.min():.2f}, {transformed_data.max():.2f}]")
        
        return df
    
    def inverse_transform(self, values, original_column_name=None, group_ids=None):
        """Reverse the log transformation to get back to original scale
        
        Args:
            values: Array of log predictions
            original_column_name: Name of original column (for reference)
            group_ids: Optional group identifiers (not used for log transform)
        """
        if not self.fitted:
            raise ValueError("Transformation must be fitted before inverse transform")
        
        # Convert back from log space
        if self.handle_zeros == 'add_constant':
            original_values = np.exp(values) - self.add_constant
        elif self.handle_zeros == 'replace':
            original_values = np.exp(values)
        elif self.handle_zeros == 'remove':
            original_values = np.exp(values)
        else:
            raise ValueError(f"Unknown handle_zeros method: {self.handle_zeros}")
        
        return original_values 