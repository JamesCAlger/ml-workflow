# How to Add New Transformations 🚀

The Transform Registry pattern makes adding new transformations incredibly easy. Here's exactly how to do it:

## ✅ **Registry Implementation Complete!**

Your system now includes:
- ✅ Transform Registry Pattern implemented
- ✅ TransformationManager updated to use registry
- ✅ BoxCox transformation as working example
- ✅ All existing transformations preserved and working
- ✅ Enhanced visualization handling

## 📋 **Step-by-Step: Adding a New Transformation**

### **Step 1: Create the Transformation Class**

Create a new file in `src/transformations/your_transform.py`:

```python
import pandas as pd
import numpy as np
from .base_transform import BaseTransform

class YourTransform(BaseTransform):
    """Your custom transformation description"""
    
    def __init__(self, your_param=default_value, **kwargs):
        """Initialize your transformation"""
        super().__init__(**kwargs)
        self.your_param = your_param
        # Add other initialization
    
    def fit(self, df, column_name):
        """Learn parameters from the data"""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        # Your fitting logic here
        # Learn parameters from df[column_name]
        
        self.fitted = True
        return self
    
    def transform(self, df, column_name):
        """Apply the transformation"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        df = df.copy()
        data = df[column_name]
        
        # Your transformation logic here
        transformed_data = your_transformation_function(data)
        
        # Create new column (preserves original)
        new_column_name = f"{column_name}_your_suffix"
        df[new_column_name] = transformed_data
        
        print(f"Applied your transform: '{column_name}' -> '{new_column_name}'")
        return df
    
    def inverse_transform(self, values, original_column_name=None, group_ids=None):
        """Reverse the transformation"""
        # Your inverse logic here
        return original_values
    
    def get_metadata(self):
        """Provide transformation metadata for intelligent handling"""
        return {
            'complexity': 'simple',           # simple, complex, structural
            'visualization_scale': 'original', # original, log, normalized, difference
            'requires_groups': False,         # Does inverse need group info?
            'output_interpretation': 'same', # same, rate_of_change, normalized
            'suggested_formatter': 'decimal' # currency, percentage, decimal, scientific
        }
```

### **Step 2: Update the __init__.py**

Add your transform to `src/transformations/__init__.py`:

```python
from .your_transform import YourTransform

__all__ = ['BaseTransform', 'FirstDifference', 'LogTransform', 'BoxCoxTransform', 'YourTransform']
```

### **Step 3: Register the Transformation**

Add your transform to `src/transform_registry.py` in the `_initialize_default_transforms()` function:

```python
def _initialize_default_transforms():
    """Initialize registry with default transformations"""
    from transformations.log_transform import LogTransform
    from transformations.first_difference import FirstDifference
    from transformations.box_cox_transform import BoxCoxTransform
    from transformations.your_transform import YourTransform  # Add this
    
    registry = _global_registry
    registry.register('log_transform', LogTransform)
    registry.register('first_difference', FirstDifference)
    registry.register('box_cox_transform', BoxCoxTransform)
    registry.register('your_transform', YourTransform)  # Add this
```

### **Step 4: Use in Configuration**

Now you can use your transform in any YAML configuration:

```yaml
# data_config.yaml
transformations:
  column_transforms:
    nav:
      - name: log_transform
        params:
          add_constant: 1
      - name: your_transform        # ← Your new transform!
        params:
          your_param: some_value
      - name: first_difference
        params:
          groupby_column: "investment"
```

### **Step 5: Test Your Transform**

Run the test to verify it works:

```bash
cd models/quantile_analysis
python test_registry.py
```

## 🎯 **Real Example: BoxCox Transform**

Here's exactly how I added the BoxCox transformation:

### **File Created:** `src/transformations/box_cox_transform.py`
- ✅ Inherits from `BaseTransform`
- ✅ Implements `fit()`, `transform()`, `inverse_transform()`
- ✅ Provides intelligent `get_metadata()`
- ✅ Handles zero/negative values
- ✅ Auto-optimizes lambda parameter

### **Registry Registration:**
```python
# In _initialize_default_transforms()
registry.register('box_cox_transform', BoxCoxTransform)
```

### **Usage in Configuration:**
```yaml
nav:
  - name: box_cox_transform
    params:
      optimize_lambda: true
      handle_zeros: "add_constant"
```

### **Result:**
- ✅ `nav` → `nav_boxcox` transformation chain
- ✅ Automatic inverse transformation for predictions
- ✅ Intelligent visualization formatting
- ✅ **Zero changes to core TransformationManager!**

## 🚀 **Advanced Examples**

### **Standardization Transform**
```python
class StandardizeTransform(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = None
        self.std = None
    
    def fit(self, df, column_name):
        data = df[column_name]
        self.mean = data.mean()
        self.std = data.std()
        self.fitted = True
        return self
    
    def transform(self, df, column_name):
        df = df.copy()
        new_col = f"{column_name}_standardized"
        df[new_col] = (df[column_name] - self.mean) / self.std
        return df
    
    def inverse_transform(self, values, original_column_name=None, group_ids=None):
        return values * self.std + self.mean
```

### **Rolling Average Transform**
```python
class RollingAverageTransform(BaseTransform):
    def __init__(self, window=5, **kwargs):
        super().__init__(**kwargs)
        self.window = window
    
    def fit(self, df, column_name):
        self.fitted = True
        return self
    
    def transform(self, df, column_name):
        df = df.copy()
        new_col = f"{column_name}_rolling_{self.window}"
        df[new_col] = df[column_name].rolling(window=self.window).mean()
        return df
    
    def get_metadata(self):
        return {
            'complexity': 'simple',
            'visualization_scale': 'original',
            'requires_groups': False,
            'output_interpretation': 'smoothed',
            'suggested_formatter': 'currency'
        }
```

## 💡 **Benefits of Registry Pattern**

### **Before (Hardcoded):**
❌ Must modify `TransformationManager` for each new transform  
❌ Violates Open/Closed Principle  
❌ Risk of breaking existing functionality  
❌ Difficult to test new transforms in isolation  

### **After (Registry Pattern):**
✅ Add transforms with **zero core code changes**  
✅ Plugin-style architecture  
✅ Easy testing and validation  
✅ Automatic error handling with helpful messages  
✅ Discoverable via `list_transforms()`  

## 🧪 **Testing Your New Transforms**

The test script automatically detects and tests new transforms:

```bash
python test_registry.py
```

This will:
- ✅ List all available transformations
- ✅ Test creation of each transform
- ✅ Show metadata for each transform
- ✅ Demonstrate configuration examples
- ✅ Test error handling

## 🎯 **Best Practices**

1. **Always inherit from `BaseTransform`**
2. **Implement all required methods** (`fit`, `transform`, `inverse_transform`)
3. **Provide meaningful `get_metadata()`** for intelligent visualization
4. **Handle edge cases** (zeros, negatives, missing values)
5. **Preserve original columns** by creating new ones
6. **Add helpful print statements** for debugging
7. **Test thoroughly** before using in experiments

## 🚀 **Ready to Use!**

Your system now supports:
- ✅ **Infinite extensibility** - add any transformation without core changes
- ✅ **Backwards compatibility** - all existing experiments work unchanged
- ✅ **Smart visualization** - automatic formatting based on transform metadata
- ✅ **Robust error handling** - helpful messages when things go wrong
- ✅ **Easy discovery** - list available transforms anytime

**The Transform Registry pattern makes your system truly modular and future-ready!** 