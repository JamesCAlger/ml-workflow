"""
Transformation Manager for handling forward and inverse transformations automatically.
The user only needs to configure forward transformations in YAML.

Enhanced with Registry Pattern for modular transformation loading.
"""
import numpy as np
from transform_registry import get_registry, create_transform, list_transforms

class TransformationManager:
    """Centralized manager for all data transformations with automatic inverse capability"""
    
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
    
    def apply_transformations(self, df, transform_config):
        """Apply configured transformations and store fitted transformers for inverse operations"""
        column_transforms = transform_config.get('column_transforms', {})
        
        for column_name, transforms in column_transforms.items():
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in dataframe, skipping transformations")
                continue
            
            print(f"\nApplying transformations to column '{column_name}':")
            
            # Track the current column name through the transformation chain
            current_column = column_name
            transformation_chain = []
            
            for i, transform_spec in enumerate(transforms):
                transform_name = transform_spec['name']
                transform_params = transform_spec.get('params', {})
                
                # Create transformation using registry pattern - NO HARDCODED CONDITIONALS!
                try:
                    transformer = create_transform(transform_name, **transform_params)
                    print(f"✅ Created {transformer.__class__.__name__} with params: {transform_params}")
                except ValueError as e:
                    print(f"❌ Error creating transformation '{transform_name}': {e}")
                    available = list_transforms()
                    print(f"📋 Available transformations: {available}")
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
                
                print(f"Stored transformer for inverse operations: {transformed_column} -> {column_name}")
            
            # Map original column to final transformed column
            if transformation_chain:
                final_column = transformation_chain[-1]['to_column']
                self.column_mappings[column_name] = final_column
                
                # Store the transformation chain for complex inverse operations
                self.transformation_chains = getattr(self, 'transformation_chains', {})
                self.transformation_chains[final_column] = transformation_chain
        
        return df
    
    def can_inverse_transform(self, column_name):
        """Check if inverse transformation is available for a column"""
        return column_name in self.fitted_transformers
    
    def inverse_transform_predictions(self, predictions, target_column, group_ids=None):
        """Automatically inverse transform predictions if possible
        
        Args:
            predictions: Dictionary of quantile predictions or single prediction array
            target_column: Name of target column to inverse transform
            group_ids: Optional array of group identifiers for group-aware transformations
        """
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
    
    def get_visualization_scale(self, target_column):
        """Get the appropriate column and scale for visualization"""
        if not self.auto_reverse_for_visualization:
            return target_column, 'transformed'
        
        if target_column in self.inverse_mappings:
            original_column = self.inverse_mappings[target_column]
            return original_column, 'original'
        
        return target_column, 'original'
    
    def get_visualization_column_info(self, target_column):
        """Get detailed information about what columns to use for visualization
        
        Returns:
            dict: Information about columns and transformation complexity
        """
        if not self.auto_reverse_for_visualization:
            return {
                'target_column': target_column,
                'actual_column': target_column,
                'needs_prediction_transform': False,
                'transform_complexity': 'none'
            }
        
        # Check if this is a chained transformation
        transformation_chains = getattr(self, 'transformation_chains', {})
        
        if target_column in transformation_chains:
            chain = transformation_chains[target_column]
            original_column = self.inverse_mappings[target_column]
            
            # For chained transformations, we need to be more careful
            # Check if the chain includes first difference
            has_first_diff = any(step['transform_name'] == 'first_difference' for step in chain)
            
            if has_first_diff:
                # First difference makes back-transformation complex
                # For visualization, we should warn and potentially use a different approach
                return {
                    'target_column': target_column,
                    'actual_column': original_column,
                    'needs_prediction_transform': True,
                    'transform_complexity': 'complex_chain',
                    'warning': 'First difference in chain makes visualization back-transformation complex',
                    'chain_info': chain
                }
            else:
                # Simple chain (e.g., just log transform)
                return {
                    'target_column': target_column,
                    'actual_column': original_column,
                    'needs_prediction_transform': True,
                    'transform_complexity': 'simple_chain',
                    'chain_info': chain
                }
        
        elif target_column in self.inverse_mappings:
            # Single transformation
            original_column = self.inverse_mappings[target_column]
            return {
                'target_column': target_column,
                'actual_column': original_column,
                'needs_prediction_transform': True,
                'transform_complexity': 'single_transform'
            }
        
        else:
            # No transformation
            return {
                'target_column': target_column,
                'actual_column': target_column,
                'needs_prediction_transform': False,
                'transform_complexity': 'none'
            }
    
    def get_target_info(self, target_column):
        """Get information about target column transformations"""
        return {
            'transformed_column': target_column,
            'original_column': self.inverse_mappings.get(target_column, target_column),
            'has_inverse': self.can_inverse_transform(target_column),
            'auto_reverse_evaluation': self.auto_reverse_for_evaluation,
            'auto_reverse_visualization': self.auto_reverse_for_visualization
        }
    
    def print_transformation_summary(self):
        """Print summary of applied transformations"""
        print("\n" + "="*60)
        print("TRANSFORMATION SUMMARY")
        print("="*60)
        
        if not self.fitted_transformers:
            print("No transformations applied")
            return
        
        # Show transformation chains if they exist
        transformation_chains = getattr(self, 'transformation_chains', {})
        if transformation_chains:
            for final_col, chain in transformation_chains.items():
                original_col = self.inverse_mappings[final_col]
                chain_description = " -> ".join([step['to_column'] for step in chain])
                transform_types = " -> ".join([step['transformer'].__class__.__name__ for step in chain])
                print(f"  {original_col} -> {chain_description} ({transform_types})")
        else:
            # Fallback to simple display
            for transformed_col, transformer in self.fitted_transformers.items():
                original_col = self.inverse_mappings[transformed_col]
                transform_type = transformer.__class__.__name__
                print(f"  {original_col} -> {transformed_col} ({transform_type})")
        
        print(f"\nAutomatic Settings:")
        print(f"  - Auto reverse for evaluation: {self.auto_reverse_for_evaluation}")
        print(f"  - Auto reverse for visualization: {self.auto_reverse_for_visualization}")
        print("="*60) 