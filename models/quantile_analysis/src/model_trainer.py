import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import QuantileRegressor
import numpy as np
import warnings
from utils import validate_config, save_models

warnings.filterwarnings('ignore')

class QuantileModelTrainer:
    """Multi-quantile model trainer supporting different algorithms"""
    
    def __init__(self, model_config):
        """Initialize trainer with model configuration"""
        self.model_config = model_config
        validate_config(model_config, ['quantiles'])
        
    def get_quantiles(self, quantile_set='default'):
        """Get quantile list from configuration"""
        return self.model_config['quantiles'][quantile_set]
    
    def train_lightgbm_quantiles(self, X_train, y_train, quantiles, feature_names):
        """Train LightGBM quantile models"""
        print("Training LightGBM quantile models...")
        
        # Get LightGBM parameters
        lgb_params = self.model_config['lightgbm'].copy()
        base_params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'boosting_type': lgb_params.get('boosting_type', 'gbdt'),
            'num_leaves': lgb_params.get('num_leaves', 31),
            'learning_rate': lgb_params.get('learning_rate', 0.05),
            'feature_fraction': lgb_params.get('feature_fraction', 0.9),
            'bagging_fraction': lgb_params.get('bagging_fraction', 0.8),
            'bagging_freq': lgb_params.get('bagging_freq', 5),
            'verbose': lgb_params.get('verbose', -1),
            'random_state': lgb_params.get('random_state', 42)
        }
        
        models = {}
        predictions_train = {}
        feature_importance = {}
        
        for quantile in quantiles:
            print(f"Training LightGBM model for quantile {quantile}...")
            
            # Set quantile-specific parameter
            params = base_params.copy()
            params['alpha'] = quantile
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=lgb_params.get('num_boost_round', 100),
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Store model and predictions
            models[quantile] = model
            predictions_train[quantile] = model.predict(X_train)
            
            # Store feature importance
            importance_values = model.feature_importance(importance_type='gain')
            feature_importance[quantile] = dict(zip(feature_names, importance_values))
        
        return models, predictions_train, feature_importance
    
    def train_xgboost_quantiles(self, X_train, y_train, quantiles, feature_names):
        """Train XGBoost quantile models"""
        print("Training XGBoost quantile models...")
        
        # Get XGBoost parameters
        xgb_params = self.model_config['xgboost'].copy()
        base_params = {
            'objective': 'reg:quantileerror',
            'tree_method': xgb_params.get('tree_method', 'hist'),
            'max_depth': xgb_params.get('max_depth', 6),
            'learning_rate': xgb_params.get('learning_rate', 0.05),
            'subsample': xgb_params.get('subsample', 0.8),
            'colsample_bytree': xgb_params.get('colsample_bytree', 0.9),
            'random_state': xgb_params.get('random_state', 42)
        }
        
        models = {}
        predictions_train = {}
        feature_importance = {}
        
        for quantile in quantiles:
            print(f"Training XGBoost model for quantile {quantile}...")
            
            # Set quantile-specific parameter
            params = base_params.copy()
            params['quantile_alpha'] = quantile
            
            # Train model
            model = xgb.XGBRegressor(
                **params,
                n_estimators=xgb_params.get('n_estimators', 100)
            )
            
            model.fit(X_train, y_train)
            
            # Store model and predictions
            models[quantile] = model
            predictions_train[quantile] = model.predict(X_train)
            
            # Store feature importance
            importance_values = model.feature_importances_
            feature_importance[quantile] = dict(zip(feature_names, importance_values))
        
        return models, predictions_train, feature_importance
    
    def train_sklearn_quantiles(self, X_train, y_train, quantiles, feature_names):
        """Train scikit-learn quantile regression models"""
        print("Training scikit-learn quantile models...")
        
        # Get sklearn parameters
        sklearn_params = self.model_config['sklearn_quantile'].copy()
        
        models = {}
        predictions_train = {}
        feature_importance = {}
        
        for quantile in quantiles:
            print(f"Training sklearn quantile model for quantile {quantile}...")
            
            # Train model
            model = QuantileRegressor(
                quantile=quantile,
                solver=sklearn_params.get('solver', 'highs'),
                alpha=sklearn_params.get('alpha', 0.1),
                fit_intercept=sklearn_params.get('fit_intercept', True)
            )
            
            model.fit(X_train, y_train)
            
            # Store model and predictions
            models[quantile] = model
            predictions_train[quantile] = model.predict(X_train)
            
            # Store feature importance (coefficients for linear models)
            if hasattr(model, 'coef_'):
                feature_importance[quantile] = dict(zip(feature_names, np.abs(model.coef_)))
            else:
                feature_importance[quantile] = {f: 0 for f in feature_names}
        
        return models, predictions_train, feature_importance
    
    def train_quantile_models(self, X_train, y_train, model_type, feature_names, quantile_set='default'):
        """Train quantile models using specified algorithm"""
        quantiles = self.get_quantiles(quantile_set)
        
        print(f"Training {model_type} models for quantiles: {quantiles}")
        print(f"Training data shape: {X_train.shape}")
        print(f"Feature names: {feature_names}")
        
        if model_type == 'lightgbm':
            models, predictions_train, feature_importance = self.train_lightgbm_quantiles(
                X_train, y_train, quantiles, feature_names
            )
        elif model_type == 'xgboost':
            models, predictions_train, feature_importance = self.train_xgboost_quantiles(
                X_train, y_train, quantiles, feature_names
            )
        elif model_type == 'sklearn_quantile':
            models, predictions_train, feature_importance = self.train_sklearn_quantiles(
                X_train, y_train, quantiles, feature_names
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print("Model training completed!")
        
        return {
            'models': models,
            'predictions_train': predictions_train,
            'feature_importance': feature_importance,
            'quantiles': quantiles,
            'model_type': model_type,
            'feature_names': feature_names
        }
    
    def generate_predictions(self, models, X_data):
        """Generate predictions for new data using trained models"""
        predictions = {}
        
        for quantile, model in models.items():
            predictions[quantile] = model.predict(X_data)
        
        return predictions
    
    def save_trained_models(self, training_results, output_path):
        """Save trained models to disk"""
        models = training_results['models']
        save_models(models, output_path)
        
        # Also save model metadata
        metadata = {
            'quantiles': training_results['quantiles'],
            'model_type': training_results['model_type'],
            'feature_names': training_results['feature_names']
        }
        
        import json
        import os
        metadata_path = os.path.join(output_path, "models", "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to: {output_path}/models/")
    
    def get_feature_importance_summary(self, feature_importance, feature_names):
        """Generate feature importance summary across all quantiles"""
        
        # Calculate average importance across quantiles
        avg_importance = {}
        for feature in feature_names:
            importance_values = [feature_importance[q][feature] for q in feature_importance.keys()]
            avg_importance[feature] = np.mean(importance_values)
        
        # Calculate relative importance
        total_importance = sum(avg_importance.values())
        relative_importance = {
            feature: (importance / total_importance) * 100 
            for feature, importance in avg_importance.items()
        }
        
        # Create summary
        import pandas as pd
        importance_summary = pd.DataFrame({
            'Feature': feature_names,
            'Average_Importance': [avg_importance[f] for f in feature_names],
            'Relative_Importance_Pct': [relative_importance[f] for f in feature_names]
        }).sort_values('Average_Importance', ascending=False)
        
        return importance_summary 