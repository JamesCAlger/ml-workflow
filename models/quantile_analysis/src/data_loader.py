import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import get_absolute_path, validate_config
from transformation_manager import TransformationManager

class DataLoader:
    """Data loading and preprocessing for quantile analysis"""
    
    def __init__(self, data_config, experiment_config):
        """Initialize DataLoader with configuration"""
        self.data_config = data_config
        self.experiment_config = experiment_config
        self.label_encoders = {}
        
        # Initialize transformation manager
        self.transformation_manager = TransformationManager(data_config)
        
        # Validate required config keys
        validate_config(data_config, ['data_sources', 'preprocessing'])
        validate_config(experiment_config, ['targets', 'covariates'])
    
    def load_raw_data(self):
        """Load data from Excel file"""
        data_source = self.data_config['data_sources']['primary']
        file_path = get_absolute_path(data_source['file_path'])
        
        print(f"Loading data from: {file_path}")
        
        df = pd.read_excel(
            file_path, 
            sheet_name=data_source['sheet_name']
        )
        
        # Drop columns that are not needed (keep proceeds for potential future use)
        if 'proceeds' in df.columns:
            df.drop(columns=['proceeds'], inplace=True)
        
        print(f"Data loaded successfully! Shape: {df.shape}")
        print(df.head())
        
        return df
    
    def preprocess_data(self, df):
        """Apply data transformations based on configuration"""
        preprocessing = self.data_config['preprocessing']
        
        # Apply NAV transformations
        if preprocessing['nav_transformations']['invert_sign']:
            df['nav'] = -df['nav']
            print("Inverted NAV signs")
        
        if preprocessing['nav_transformations']['set_negative_to_zero']:
            df.loc[df['nav'] < 0, 'nav'] = 0
            print("Set negative NAV to zero")
        
        min_threshold = preprocessing['nav_transformations']['min_threshold']
        df.loc[(df['nav'] >= 0) & (df['nav'] <= min_threshold), 'nav'] = 0
        print(f"Set NAV <= {min_threshold} to zero")
        
        # Apply column transformations using TransformationManager
        if 'transformations' in preprocessing:
            df = self.transformation_manager.apply_transformations(df, preprocessing['transformations'])
            self.transformation_manager.print_transformation_summary()
        
        print("Data after transformations:")
        print(df.head(10))
        
        return df
    
    def get_transformation_manager(self):
        """Get the transformation manager for use by other components"""
        return self.transformation_manager
    
    def create_age_column(self, df):
        """Create age column based on investment date ranking"""
        date_col = self.data_config['data_sources']['primary']['date_column']
        groupby_col = self.data_config['preprocessing']['age_settings']['groupby_column']
        
        # Convert date to datetime if not already
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by investment and date
        df = df.sort_values([groupby_col, date_col])
        
        # Create age column that starts at 1 for earliest date for each investment
        df['age'] = df.groupby(groupby_col)[date_col].rank(method='dense').astype(int)
        
        print("Created age column")
        print(df[[groupby_col, date_col, 'age', 'nav']].head(15))
        
        return df
    
    def filter_data(self, df):
        """Apply data filters based on configuration"""
        filters = self.data_config.get('filters', {})
        preprocessing = self.data_config['preprocessing']
        
        original_shape = df.shape
        
        # Filter by max age
        max_age = preprocessing['age_settings']['max_age']
        df_filtered = df[df['age'] <= max_age]
        print(f"Filtered to age <= {max_age}: {df_filtered.shape}")
        
        # Remove zero disbursement if specified  
        if filters.get('remove_zero_disbursement', True):
            # For disbursement, we want to keep negative values (they represent actual disbursements)
            # Only remove exactly zero values
            df_filtered = df_filtered[df_filtered['disbursement'] != 0]
            print(f"Removed zero disbursement: {df_filtered.shape}")
        
        print(f"Original shape: {original_shape}")
        print(f"Final filtered shape: {df_filtered.shape}")
        print(f"Rows removed: {original_shape[0] - df_filtered.shape[0]}")
        
        return df_filtered
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        encoding_config = self.data_config['preprocessing']['encoding']
        strategy_col = encoding_config['strategy_column']
        method = encoding_config['method']
        
        # Get aggregation covariate column
        aggregation_config = self.data_config['preprocessing'].get('aggregation', {})
        aggregation_col = aggregation_config.get('covariate_column', 'strategy')
        
        if method == 'label_encoder':
            le = LabelEncoder()
            df['strategy_encoded'] = le.fit_transform(df[strategy_col])
            self.label_encoders['strategy'] = le
            
            # Print strategy mapping
            strategy_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Strategy encoding mapping: {strategy_mapping}")
            
            # If aggregation column is different from strategy, encode it too
            if aggregation_col != strategy_col and aggregation_col in df.columns:
                le_agg = LabelEncoder()
                df[f'{aggregation_col}_encoded'] = le_agg.fit_transform(df[aggregation_col])
                self.label_encoders[aggregation_col] = le_agg
                
                # Print aggregation covariate mapping
                agg_mapping = dict(zip(le_agg.classes_, le_agg.transform(le_agg.classes_)))
                print(f"{aggregation_col.title()} encoding mapping: {agg_mapping}")
            
        elif method == 'one_hot':
            # One-hot encoding
            strategy_dummies = pd.get_dummies(df[strategy_col], prefix='strategy')
            df = pd.concat([df, strategy_dummies], axis=1)
            print(f"Created one-hot encoding for {strategy_col}")
            
            # If aggregation column is different from strategy, encode it too
            if aggregation_col != strategy_col and aggregation_col in df.columns:
                agg_dummies = pd.get_dummies(df[aggregation_col], prefix=aggregation_col)
                df = pd.concat([df, agg_dummies], axis=1)
                print(f"Created one-hot encoding for {aggregation_col}")
        
        return df
    
    def prepare_target_variable(self, df, target_name):
        """Prepare target variable (simplified - transformations now handled in preprocessing)"""
        target_config = self.experiment_config['targets'][target_name]
        
        # Get target column (should already be transformed via preprocessing)
        target_col = target_config['column']
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {list(df.columns)}")
        
        target_data = df[target_col].copy()
        
        print(f"Target variable '{target_name}' (column: '{target_col}') statistics:")
        print(target_data.describe())
        
        return target_data
    
    def prepare_features(self, df, covariate_set_name):
        """Prepare feature matrix based on covariate configuration"""
        covariate_config = self.experiment_config['covariates'][covariate_set_name]
        features = covariate_config['features']
        
        # Validate that all features exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataframe: {missing_features}")
        
        X = df[features].copy()
        
        print(f"Prepared features from '{covariate_set_name}' set:")
        print(f"Features: {features}")
        print(f"Feature matrix shape: {X.shape}")
        
        return X, features
    
    def create_stratified_investment_split(self, X, y, df, test_size):
        """Create stratified train/test split at investment level maintaining covariate balance"""
        from sklearn.model_selection import train_test_split
        
        # Get configuration
        groupby_col = self.data_config['preprocessing']['age_settings']['groupby_column']
        aggregation_config = self.data_config['preprocessing'].get('aggregation', {})
        aggregation_col = aggregation_config.get('covariate_column', 'strategy')
        
        print(f"Creating stratified split by {aggregation_col} at {groupby_col} level...")
        
        # Get unique investments and their primary covariate class
        investment_covariate = df.groupby(groupby_col)[aggregation_col].first().reset_index()
        
        print(f"Total unique {groupby_col}s: {len(investment_covariate)}")
        print(f"Covariate distribution:")
        covariate_counts = investment_covariate[aggregation_col].value_counts()
        print(covariate_counts)
        
        # Split investments maintaining covariate balance
        investments_train, investments_test = train_test_split(
            investment_covariate[groupby_col],
            test_size=test_size,
            stratify=investment_covariate[aggregation_col],
            random_state=42
        )
        
        print(f"Training {groupby_col}s: {len(investments_train)}")
        print(f"Test {groupby_col}s: {len(investments_test)}")
        
        # Create train/test masks
        train_mask = df[groupby_col].isin(investments_train)
        test_mask = df[groupby_col].isin(investments_test)
        
        # Split the data
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        df_train = df[train_mask]
        df_test = df[test_mask]
        
        # Verify covariate balance
        print(f"\nTrain set {aggregation_col} distribution:")
        train_covariate_counts = df_train[aggregation_col].value_counts()
        print(train_covariate_counts)
        print(f"Train set proportions: {(train_covariate_counts / len(df_train)).round(3).to_dict()}")
        
        print(f"\nTest set {aggregation_col} distribution:")
        test_covariate_counts = df_test[aggregation_col].value_counts()
        print(test_covariate_counts)
        print(f"Test set proportions: {(test_covariate_counts / len(df_test)).round(3).to_dict()}")
        
        return X_train, X_test, y_train, y_test, df_train, df_test

    def create_train_test_split(self, X, y, df):
        """Create train/test split based on configuration"""
        split_config = self.data_config['preprocessing']['train_test_split']
        method = split_config['method']
        test_size = split_config['test_size']
        
        if method == 'temporal':
            # Sort by date to maintain temporal order
            date_col = self.data_config['data_sources']['primary']['date_column']
            df_sorted = df.sort_values(date_col)
            
            # Get corresponding indices for X and y
            sorted_indices = df_sorted.index
            X_sorted = X.loc[sorted_indices]
            y_sorted = y.loc[sorted_indices]
            
            # Split based on time
            split_idx = int((1 - test_size) * len(X_sorted))
            X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
            y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
            
            print(f"Created temporal split with test_size={test_size}")
            
            # Return sorted dataframes for later use
            df_train = df_sorted.iloc[:split_idx]
            df_test = df_sorted.iloc[split_idx:]
            
        elif method == 'stratified':
            # Use stratified split at investment level
            X_train, X_test, y_train, y_test, df_train, df_test = self.create_stratified_investment_split(
                X, y, df, test_size
            )
            
        elif method == 'random':
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print(f"Created random split with test_size={test_size}")
            
            # For random split, we don't maintain order
            df_train, df_test = None, None
        
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        print(f"Train set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test, df_train, df_test
    
    def prepare_experiment_data(self, experiment_name):
        """Complete data preparation pipeline for an experiment"""
        print(f"Preparing data for experiment: {experiment_name}")
        
        # Get experiment configuration
        exp_config = self.experiment_config['experiments'][experiment_name]
        target_name = exp_config['target']
        covariate_set = exp_config['covariates']
        
        # Load and preprocess data
        df = self.load_raw_data()
        df = self.preprocess_data(df)
        df = self.create_age_column(df)
        df = self.filter_data(df)
        df = self.encode_categorical_features(df)
        
        # Prepare target and features
        y = self.prepare_target_variable(df, target_name)
        X, feature_names = self.prepare_features(df, covariate_set)
        
        # Create train/test split
        X_train, X_test, y_train, y_test, df_train, df_test = self.create_train_test_split(X, y, df)
        
        # Preserve group information for log-difference back-transformation
        groupby_col = self.data_config['preprocessing']['age_settings']['groupby_column']
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'df_train': df_train,
            'df_test': df_test,
            'df_full': df,
            'label_encoders': self.label_encoders,
            # Add group information for back-transformation
            'group_info': {
                'groupby_column': groupby_col,
                'train_groups': df_train[groupby_col].values if df_train is not None else None,
                'test_groups': df_test[groupby_col].values if df_test is not None else None,
                'full_groups': df[groupby_col].values
            }
        } 