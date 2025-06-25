"""
Enhanced Transformation Manager using Registry Pattern

Demonstrates how to refactor the existing TransformationManager
to use the Transform Registry pattern for better modularity.
"""
import numpy as np
from .transform_registry import get_registry, create_transform, list_transforms


class RegistryBasedTransformationManager:
    """Enhanced Centralized manager using Transform Registry pattern"""
    
    def __init__(self, data_config):
        """Initialize with data configuration"""
        self.data_config = data_config
        self.fitted_transformers = {}  # Store fitted transformer instances
        self.column_mappings = {}  # original_column -> transformed_column
        self.inverse_mappings = {}  # transformed_column -> original_column
        self.auto_reverse_for_evaluation = True
        self.auto_reverse_for_visualization = True
        self.registry = get_registry()  # Get the global registry
        
        # Parse configuration
        self._parse_config()
    
    def _parse_config(self):
        """Parse transformation configuration from data_config"""
        transform_config = self.data_config.get('preprocessing', {}).get('transformations', {})
        
        # Set automatic reverse preferences (with smart defaults)
        self.auto_reverse_for_evaluation = transform_config.get('auto_reverse_for_evaluation', True)
        self.auto_reverse_for_visualization = transform_config.get('auto_reverse_for_visualization', True)
    
    def list_available_transforms(self):
        """List all available transformations"""
        return self.registry.list_available()
    
    def apply_transformations(self, df, transform_config):
        """Apply configured transformations using registry pattern"""
        column_transforms = transform_config.get('column_transforms', {})
        
        for column_name, transforms in column_transforms.items():
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in dataframe, skipping transformations")
                continue
            
            print(f"\nApplying transformations to column '{column_name}':")
            print(f"Available transforms: {self.list_available_transforms()}")
            
            # Track the current column name through the transformation chain
            current_column = column_name
            transformation_chain = []
            
            for i, transform_spec in enumerate(transforms):
                transform_name = transform_spec['name']
                transform_params = transform_spec.get('params', {})
                
                # Use registry to create transformation - NO HARDCODED CONDITIONALS!
                try:
                    transformer = create_transform(transform_name, **transform_params)
                except ValueError as e:
                    print(f"Error: {e}")
                    available = self.list_available_transforms()
                    print(f"Available transformations: {available}")
                    continue
                
                # Apply transformation to current column
                df = transformer.fit_transform(df, current_column)
                
                # Determine the output column name
                transform_suffix = transform_name.replace('_transform', '')
                if i == 0:
                    # First transformation: column_name -> column_name_suffix
                    transformed_column = f"{column_name}_{transform_suffix}"
                else:
                    # Subsequent transformations: add suffix to existing name
                    transformed_column = f"{current_column}_{transform_suffix}"
                
                # Store fitted transformer for inverse operations
                self.fitted_transformers[transformed_column] = transformer
                transformation_chain.append({
                    'transformer': transformer,
                    'from_column': current_column,
                    'to_column': transformed_column,
                    'transform_name': transform_name
                })
                
                # Store inverse mapping for each intermediate transformation step
                self.inverse_mappings[transformed_column] = column_name
                
                # Update current column for next transformation in chain
                current_column = transformed_column
                
                print(f"✅ Created {transformer.__class__.__name__}: {transformed_column} -> {column_name}")
            
            # Map original column to final transformed column
            if transformation_chain:
                final_column = transformation_chain[-1]['to_column']
                self.column_mappings[column_name] = final_column
                
                # Store the transformation chain for complex inverse operations
                self.transformation_chains = getattr(self, 'transformation_chains', {})
                self.transformation_chains[final_column] = transformation_chain
        
        return df
    
    # [Rest of the methods remain the same as in original TransformationManager]
    def can_inverse_transform(self, column_name):
        """Check if inverse transformation is available for a column"""
        return column_name in self.fitted_transformers
    
    def inverse_transform_predictions(self, predictions, target_column, group_ids=None):
        """Automatically inverse transform predictions if possible"""
        if not self.auto_reverse_for_evaluation:
            return predictions
            
        if not self.can_inverse_transform(target_column):
            print(f"No inverse transformation available for '{target_column}', returning original predictions")
            return predictions
        
        transformer = self.fitted_transformers[target_column]
        original_column = self.inverse_mappings[target_column]
        
        print(f"Applying inverse transformation: {target_column} -> {original_column}")
        
        # Apply standard inverse transformation
        transformer_type = transformer.__class__.__name__
        print(f"Applying standard inverse transformation for {transformer_type}")
        
        # Handle both single predictions and dictionary of quantile predictions
        if isinstance(predictions, dict):
            # Multiple quantiles
            inverse_predictions = {}
            for quantile, pred_values in predictions.items():
                inverse_predictions[quantile] = transformer.inverse_transform(
                    pred_values, original_column, group_ids=group_ids)
            return inverse_predictions
        else:
            # Single prediction array
            return transformer.inverse_transform(predictions, original_column, group_ids=group_ids)


# Example of adding a new transformation without modifying core code
class StandardizeTransform:
    """Example new transformation - standardization (z-score)"""
    
    def __init__(self, **kwargs):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit_transform(self, df, column_name):
        """Fit and transform in one step"""
        data = df[column_name]
        self.mean = data.mean()
        self.std = data.std()
        self.fitted = True
        
        # Create standardized column
        new_column = f"{column_name}_standardized"
        df[new_column] = (data - self.mean) / self.std
        
        print(f"Standardized '{column_name}' -> '{new_column}' (mean={self.mean:.2f}, std={self.std:.2f})")
        return df
    
    def inverse_transform(self, values, original_column_name=None, group_ids=None):
        """Convert standardized values back to original scale"""
        return values * self.std + self.mean


# Register the new transformation - ZERO changes to core code needed!
def register_standardize_transform():
    """Register the standardize transformation"""
    from .transform_registry import register_transform
    register_transform('standardize', StandardizeTransform)
    print("📊 Registered StandardizeTransform - ready to use!")


# Usage example:
def demonstrate_registry_usage():
    """Show how easy it is to add and use new transformations"""
    
    # 1. Register new transformation
    register_standardize_transform()
    
    # 2. Use in YAML configuration (no code changes needed):
    yaml_example = """
    transformations:
      column_transforms:
        nav:
          - name: log_transform
            params:
              add_constant: 1
          - name: standardize          # NEW TRANSFORM!
            params: {}
          - name: first_difference
            params:
              groupby_column: "investment"
    """
    
    print("New transformation chain would create:")
    print("nav → nav_log → nav_log_standardized → nav_log_standardized_diff")
    print("\nNo modifications to TransformationManager needed! 🎉")


if __name__ == "__main__":
    demonstrate_registry_usage() 