from abc import ABC, abstractmethod
import pandas as pd

class BaseTransform(ABC):
    """Base class for all data transformations"""
    
    def __init__(self, **kwargs):
        """Initialize transformation with parameters"""
        self.params = kwargs
        self.fitted = False
        self.transform_params = {}
    
    @abstractmethod
    def fit(self, df, column_name):
        """Fit the transformation on the data (learn parameters if needed)"""
        pass
    
    @abstractmethod
    def transform(self, df, column_name):
        """Apply the transformation to the data"""
        pass
    
    @abstractmethod
    def inverse_transform(self, values, original_column_name=None, group_ids=None):
        """Reverse the transformation to get back to original scale"""
        pass
    
    def fit_transform(self, df, column_name):
        """Fit and transform in one step"""
        self.fit(df, column_name)
        return self.transform(df, column_name)
    
    def get_transform_name(self):
        """Get the name of this transformation"""
        return self.__class__.__name__.lower()
    
    def get_params(self):
        """Get transformation parameters"""
        return self.params
    
    def get_metadata(self):
        """Get transformation metadata for intelligent handling
        
        Override this method in subclasses to provide specific metadata
        """
        return {
            'complexity': 'simple',           # simple, complex, structural
            'visualization_scale': 'original', # original, log, normalized, difference
            'requires_groups': False,         # Does inverse need group info?
            'output_interpretation': 'same', # same, rate_of_change, normalized
            'suggested_formatter': 'currency' # currency, percentage, decimal, scientific
        } 