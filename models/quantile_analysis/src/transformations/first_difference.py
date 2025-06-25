import pandas as pd
import numpy as np
from .base_transform import BaseTransform

class FirstDifference(BaseTransform):
    """Calculate first difference of a column within groups"""
    
    def __init__(self, groupby_column='investment', **kwargs):
        """
        Initialize first difference transformation
        
        Args:
            groupby_column: Column to group by when calculating differences
        """
        super().__init__(**kwargs)
        self.groupby_column = groupby_column
    
    def fit(self, df, column_name):
        """Fit the transformation (no parameters to learn for first difference)"""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        if self.groupby_column not in df.columns:
            raise ValueError(f"Groupby column '{self.groupby_column}' not found in dataframe")
        
        self.fitted = True
        return self
    
    def transform(self, df, column_name):
        """Apply first difference transformation"""
        if not self.fitted:
            raise ValueError("Transformation must be fitted before transform")
        
        df = df.copy()
        
        # Sort by groupby column and ensure proper ordering
        # Assuming there's a date or age column for proper ordering
        sort_columns = [self.groupby_column]
        if 'date' in df.columns:
            sort_columns.append('date')
        elif 'age' in df.columns:
            sort_columns.append('age')
        
        df = df.sort_values(sort_columns)
        
        # Calculate first difference within groups
        new_column_name = f"{column_name}_diff"
        df[new_column_name] = df.groupby(self.groupby_column)[column_name].diff()
        
        # Remove rows with NaN (first observation in each group)
        df = df.dropna(subset=[new_column_name])
        
        print(f"Applied first difference to '{column_name}' -> '{new_column_name}'")
        print(f"Original column '{column_name}' preserved alongside '{new_column_name}'")
        print(f"Rows after transformation: {len(df)}")
        
        return df
    
    def inverse_transform(self, values, original_column_name=None, group_ids=None):
        """Reverse the first difference transformation (cumulative sum)
        
        Args:
            values: Array of first difference predictions
            original_column_name: Name of original column (for reference)
            group_ids: Optional group identifiers (basic support)
        """
        if not self.fitted:
            raise ValueError("Transformation must be fitted before inverse transform")
        
        # Note: This is a simplified inverse - in practice, you'd need the initial values
        # for each group to properly reconstruct the original series
        print("Warning: First difference inverse transformation requires initial values for proper reconstruction")
        
        if group_ids is not None:
            print("Warning: Group-aware first difference inverse not fully implemented")
        
        # Simple cumulative sum (this assumes starting from 0)
        return np.cumsum(values) 