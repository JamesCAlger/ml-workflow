# Transformation System Guide

## Overview

The enhanced transformation system now supports:
1. **Preserving original columns** - transformations create new columns instead of replacing originals
2. **Chained transformations** - apply multiple transformations in sequence to create complex features
3. **Automatic inverse transformations** - for back-transforming predictions to original scale

## Key Features

### 1. Column Preservation
- **Before**: `nav` → `nav_log` (original `nav` column was replaced)
- **After**: `nav` + `nav_log` (both columns exist in the dataframe)

### 2. Transformation Chaining
You can now chain multiple transformations on the same column:

```yaml
nav:
  - name: log_transform
    params:
      add_constant: 1
      handle_zeros: "add_constant"
  - name: first_difference
    params:
      groupby_column: "investment"
```

This creates:
- `nav` (original column)
- `nav_log` (after log transformation)
- `nav_log_diff` (after first difference of log values)

### 3. Available Transformations

#### Log Transform
```yaml
column_name:
  - name: log_transform
    params:
      add_constant: 1                    # Add before log (default: 1)
      handle_zeros: "add_constant"       # How to handle zeros/negatives
```

Options for `handle_zeros`:
- `"add_constant"`: Add constant to all values
- `"replace"`: Replace zeros/negatives with small positive value
- `"remove"`: Remove zero/negative observations

#### First Difference
```yaml
column_name:
  - name: first_difference
    params:
      groupby_column: "investment"       # Group by this column for differences
```

## Example Configurations

### Simple Log Transform (creates nav + nav_log)
```yaml
transformations:
  column_transforms:
    nav:
      - name: log_transform
        params:
          add_constant: 1
          handle_zeros: "add_constant"
```

### Log-Differenced Series (creates nav + nav_log + nav_log_diff)
```yaml
transformations:
  column_transforms:
    nav:
      - name: log_transform
        params:
          add_constant: 1
          handle_zeros: "add_constant"
      - name: first_difference
        params:
          groupby_column: "investment"
```

### Multiple Column Transformations
```yaml
transformations:
  column_transforms:
    disbursement:
      - name: log_transform
        params:
          add_constant: 1
          handle_zeros: "add_constant"
    
    nav:
      - name: log_transform
        params:
          add_constant: 1
          handle_zeros: "add_constant"
      - name: first_difference
        params:
          groupby_column: "investment"
```

## Target Configuration

After defining transformations, configure your targets in `experiment_config.yaml`:

```yaml
targets:
  nav_log:
    column: nav_log
    description: "Log-transformed NAV amounts"
  
  nav_log_diff:
    column: nav_log_diff
    description: "Log-differenced NAV amounts - changes in log NAV"

experiments:
  nav_log_diff_baseline:
    target: nav_log_diff
    covariates: base_set
    model_type: lightgbm
    quantiles: default
```

## Running Experiments

```bash
# Run log-differenced NAV experiment
python run_experiment.py --experiment nav_log_diff_baseline

# Run standard log NAV experiment
python run_experiment.py --experiment nav_baseline
```

## Technical Notes

### Inverse Transformations
The system automatically handles inverse transformations for evaluation and visualization:
- Log transforms are reversed using `exp(x) - constant`
- First differences require initial values for proper reconstruction (simplified implementation)

### Column Naming Convention
- Single transform: `{original_name}_{transform_suffix}`
- Chained transforms: `{previous_name}_{next_suffix}`

Examples:
- `nav` → `nav_log`
- `nav_log` → `nav_log_diff`

### Data Requirements
- **Log Transform**: Handles zeros/negatives based on configuration
- **First Difference**: Requires grouping column and removes first observation per group

## Benefits

1. **Flexibility**: Keep original data while creating derived features
2. **Analysis**: Compare original vs. transformed series performance
3. **Complex Features**: Create sophisticated features like log-returns, growth rates
4. **Interpretability**: Easier to understand and validate transformations 