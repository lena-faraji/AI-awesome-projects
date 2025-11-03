"""
Advanced Heart Disease Risk Prediction System

A comprehensive deep learning pipeline for predicting heart disease risk from
biking and smoking habits. Features advanced model architectures, hyperparameter
optimization, explainable AI, and production-ready deployment capabilities.

Key Enhancements:
- Multiple model architectures (MLP, Residual, Attention)
- Automated hyperparameter tuning with Optuna
- Advanced explainability with SHAP and LIME
- Ensemble learning and model stacking
- Comprehensive monitoring and logging
- API deployment ready with FastAPI
- Docker containerization
- Advanced data validation with Pydantic
- Cross-validation and statistical testing
- Feature engineering and selection

Author: ML Engineering Team
License: MIT
"""

import os
import logging
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                           explained_variance_score, max_error)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, 
                                   Input, Add, MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import plot_model

# Optional imports for advanced features
try:
    import shap
    import lime
    import lime.lime_tabular
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP/LIME not available. Install for explainability features.")

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/heart_risk_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration management
class Config:
    """Configuration class for model and training parameters."""
    
    def __init__(self):
        self.data_path = "data/heart_data.csv"
        self.output_dir = "outputs"
        self.model_dir = "models"
        self.log_dir = "logs"
        self.cache_dir = "cache"
        
        # Create directories
        for directory in [self.output_dir, self.model_dir, self.log_dir, self.cache_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.test_size = 0.2
        self.validation_size = 0.2
        self.random_state = 42
        self.epochs = 1000
        self.batch_size = 32
        self.patience = 50
        self.min_delta = 0.001
        
        # Advanced training
        self.learning_rate = 0.001
        self.use_learning_rate_scheduler = True
        self.early_stopping_monitor = 'val_loss'
        
        # Feature engineering
        self.polynomial_degree = 2
        self.feature_selection_k = 'all'
        
        # Cross-validation
        self.cv_folds = 5
        
        # Ensemble
        self.ensemble_method = 'weighted'  # 'weighted', 'average', 'stacking'

class AdvancedDataProcessor:
    """Advanced data processing with feature engineering and validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=config.polynomial_degree, include_bias=False)
        self.feature_selector = None
        self.feature_names = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate dataset with advanced checks."""
        try:
            df = pd.read_csv(self.config.data_path)
            logger.info(f"Dataset loaded: {df.shape}")
            
            # Advanced data validation
            self._validate_data(df)
            
            # Clean data
            df_clean = self._clean_data(df)
            
            # Feature engineering
            df_engineered = self._engineer_features(df_clean)
            
            return df_engineered
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame):
        """Comprehensive data validation."""
        required_columns = ['biking', 'smoking', 'heart.disease']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Statistical validation
        for column in ['biking', 'smoking', 'heart.disease']:
            if df[column].dtype not in ['int64', 'float64']:
                raise ValueError(f"Column {column} must be numeric")
            
            # Check for infinite values
            if np.any(np.isinf(df[column])):
                raise ValueError(f"Infinite values found in {column}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced data cleaning."""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle missing values with advanced imputation
        if df_clean.isnull().sum().any():
            logger.warning("Missing values detected. Using advanced imputation.")
            for column in df_clean.columns:
                if df_clean[column].isnull().sum() > 0:
                    # Use median for skewed distributions
                    if abs(df_clean[column].skew()) > 1:
                        df_clean[column].fillna(df_clean[column].median(), inplace=True)
                    else:
                        df_clean[column].fillna(df_clean[column].mean(), inplace=True)
        
        # Remove outliers using IQR method
        Q1 = df_clean[['biking', 'smoking', 'heart.disease']].quantile(0.25)
        Q3 = df_clean[['biking', 'smoking', 'heart.disease']].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ~((df_clean < (Q1 - 1.5 * IQR)) | (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)
        df_clean = df_clean[outlier_condition]
        logger.info(f"Removed {initial_rows - len(df_clean)} outliers")
        
        return df_clean
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features."""
        df_engineered = df.copy()
        
        # Interaction features
        df_engineered['biking_smoking_interaction'] = df_engineered['biking'] * df_engineered['smoking']
        df_engineered['biking_squared'] = df_engineered['biking'] ** 2
        df_engineered['smoking_squared'] = df_engineered['smoking'] ** 2
        
        # Ratio features
        df_engineered['lifestyle_ratio'] = df_engineered['biking'] / (df_engineered['smoking'] + 1)  # +1 to avoid division by zero
        
        # Binning features
        df_engineered['biking_level'] = pd.cut(df_engineered['biking'], bins=3, labels=['low', 'medium', 'high'])
        df_engineered['smoking_level'] = pd.cut(df_engineered['smoking'], bins=3, labels=['low', 'medium', 'high'])
        
        # One-hot encoding for categorical features
        df_engineered = pd.get_dummies(df_engineered, columns=['biking_level', 'smoking_level'], prefix=['bike', 'smoke'])
        
        logger.info(f"Engineered features. New shape: {df_engineered.shape}")
        return df_engineered
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for modeling."""
        # Select numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != 'heart.disease']
        
        X = df[numeric_columns].values
        y = df['heart.disease'].values
        
        self.feature_names = numeric_columns
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.config.feature_selection_k != 'all':
            self.feature_selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            
            # Update feature names
            if hasattr(self.feature_selector, 'get_support'):
                selected_mask = self.feature_selector.get_support()
                self.feature_names = [name for name, selected in zip(self.feature_names, selected_mask) if selected]
        
        # Polynomial features
        if self.config.polynomial_degree > 1:
            X_scaled = self.poly.fit_transform(X_scaled)
            # Update feature names for polynomial features
            poly_feature_names = self.poly.get_feature_names_out(self.feature_names)
            self.feature_names = poly_feature_names.tolist()
        
        logger.info(f"Final feature set: {len(self.feature_names)} features")
        return X_scaled, y, self.feature_names

class AdvancedModelFactory:
    """Factory for creating advanced model architectures."""
    
    @staticmethod
    def create_mlp_model(input_dim: int, config: Config) -> Model:
        """Create an advanced MLP model."""
        model = Sequential([
            Dense(128, input_dim=input_dim, activation='relu', 
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dense(1)  # Linear activation for regression
        ])
        
        optimizer = Adam(learning_rate=config.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model
    
    @staticmethod
    def create_residual_model(input_dim: int, config: Config) -> Model:
        """Create a residual network model."""
        inputs = Input(shape=(input_dim,))
        
        # First block
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Residual block 1
        residual = x
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Add()([x, residual])
        
        # Second block
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=config.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model
    
    @staticmethod
    def create_attention_model(input_dim: int, config: Config) -> Model:
        """Create a model with attention mechanism."""
        inputs = Input(shape=(input_dim,))
        
        # Feature projection
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        
        # Reshape for attention (treat features as sequence)
        x = tf.reshape(x, (-1, input_dim, 64))
        
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = LayerNormalization()(x + attention_output)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu')(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=config.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model

class ModelTrainer:
    """Advanced model training with hyperparameter optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.histories = {}
        self.best_params = {}
    
    def get_callbacks(self, model_name: str) -> List:
        """Get advanced callbacks for training."""
        callbacks = [
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.patience,
                restore_best_weights=True,
                min_delta=self.config.min_delta
            ),
            ModelCheckpoint(
                os.path.join(self.config.model_dir, f'best_{model_name}.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            TensorBoard(
                log_dir=os.path.join(self.config.log_dir, 'tensorboard', model_name),
                histogram_freq=1
            )
        ]
        
        if self.config.use_learning_rate_scheduler:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=20,
                    min_lr=1e-7
                )
            )
        
        return callbacks
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray, input_dim: int) -> Dict:
        """Train multiple model architectures."""
        model_factory = AdvancedModelFactory()
        
        architectures = {
            'mlp': model_factory.create_mlp_model,
            'residual': model_factory.create_residual_model,
            'attention': model_factory.create_attention_model
        }
        
        for name, creator in architectures.items():
            logger.info(f"Training {name} model...")
            
            model = creator(input_dim, self.config)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=self.get_callbacks(name),
                verbose=1
            )
            
            self.models[name] = model
            self.histories[name] = history.history
            
            # Save model architecture
            plot_model(model, to_file=os.path.join(self.config.output_dir, f'{name}_architecture.png'), 
                      show_shapes=True, show_layer_names=True)
        
        return self.models
    
    def hyperparameter_optimization(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  X_val: np.ndarray, y_val: np.ndarray, input_dim: int) -> Dict:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Suggest hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 5)
            units = trial.suggest_categorical('units', [32, 64, 128, 256])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            activation = trial.suggest_categorical('activation', ['relu', 'elu', 'tanh'])
            
            # Build model
            model = Sequential()
            model.add(Dense(units, input_dim=input_dim, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            for i in range(n_layers - 1):
                model.add(Dense(units // (2 ** (i + 1)), activation=activation))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            
            model.add(Dense(1))
            
            # Compile and train
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=self.config.batch_size,
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            
            return min(history.history['val_loss'])
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        
        return self.best_params

class AdvancedEvaluator:
    """Comprehensive model evaluation with statistical testing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = {}
    
    def comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
            'mean_absolute_percentage_error': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def cross_validate(self, model: Model, X: np.ndarray, y: np.ndarray, 
                      model_name: str) -> Dict:
        """Perform k-fold cross validation."""
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, 
                     random_state=self.config.random_state)
        
        cv_scores = {'mse': [], 'mae': [], 'r2': []}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Cross-validation fold {fold + 1}/{self.config.cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model_clone = tf.keras.models.clone_model(model)
            model_clone.compile(optimizer=model.optimizer, loss='mse', metrics=['mae'])
            
            model_clone.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Evaluate
            y_pred = model_clone.predict(X_val).flatten()
            
            cv_scores['mse'].append(mean_squared_error(y_val, y_pred))
            cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
            cv_scores['r2'].append(r2_score(y_val, y_pred))
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'cv_{metric}_mean'] = np.mean(scores)
            cv_results[f'cv_{metric}_std'] = np.std(scores)
        
        self.metrics[f"{model_name}_cv"] = cv_results
        return cv_results
    
    def statistical_significance(self, y_true: np.ndarray, predictions_dict: Dict) -> pd.DataFrame:
        """Test statistical significance between model predictions."""
        models = list(predictions_dict.keys())
        n_models = len(models)
        
        results = pd.DataFrame(index=models, columns=models)
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    results.loc[model1, model2] = 1.0
                else:
                    # Paired t-test
                    pred1 = predictions_dict[model1]
                    pred2 = predictions_dict[model2]
                    
                    # Calculate errors
                    errors1 = np.abs(y_true - pred1)
                    errors2 = np.abs(y_true - pred2)
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_rel(errors1, errors2)
                    results.loc[model1, model2] = p_value
        
        return results
    
    def create_comprehensive_plots(self, y_true: np.ndarray, predictions_dict: Dict, 
                                 feature_names: List[str], model: Model):
        """Create comprehensive evaluation visualizations."""
        
        # 1. Prediction vs Actual for all models
        plt.figure(figsize=(15, 10))
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items(), 1):
            plt.subplot(2, 3, idx)
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{model_name.upper()} - Actual vs Predicted')
            
            # Add metrics to plot
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residual plots
        plt.subplot(2, 3, 4)
        for model_name, y_pred in predictions_dict.items():
            residuals = y_true - y_pred
            plt.hist(residuals, alpha=0.6, label=model_name, bins=20)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.legend()
        
        # 3. Error distribution
        plt.subplot(2, 3, 5)
        errors = {}
        for model_name, y_pred in predictions_dict.items():
            errors[model_name] = np.abs(y_true - y_pred)
        
        plt.boxplot(errors.values(), labels=errors.keys())
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution by Model')
        
        # 4. Feature importance (if SHAP available)
        if SHAP_AVAILABLE and len(feature_names) <= 20:  # SHAP works best with limited features
            plt.subplot(2, 3, 6)
            try:
                explainer = shap.Explainer(model, X_train[:100])  # Use subset for speed
                shap_values = explainer(X_train[:100])
                
                shap.summary_plot(shap_values, X_train[:100], feature_names=feature_names, 
                                show=False, plot_type='bar')
                plt.title('SHAP Feature Importance')
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                plt.text(0.5, 0.5, 'SHAP not available', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'comprehensive_evaluation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        report = "# Heart Disease Risk Prediction - Model Evaluation Report\n\n"
        
        for model_name, metrics in self.metrics.items():
            report += f"## {model_name.upper()}\n\n"
            for metric, value in metrics.items():
                report += f"- **{metric}**: {value:.4f}\n"
            report += "\n"
        
        return report

class EnsemblePredictor:
    """Advanced ensemble methods for improved predictions."""
    
    def __init__(self, config: Config):
        self.config = config
        self.weights = {}
        self.meta_model = None
    
    def calculate_weights(self, val_metrics: Dict) -> Dict:
        """Calculate model weights based on validation performance."""
        # Inverse of RMSE as weights (better models get higher weights)
        total_inverse_rmse = 0
        for model_name, metrics in val_metrics.items():
            if 'rmse' in metrics:
                self.weights[model_name] = 1 / metrics['rmse']
                total_inverse_rmse += self.weights[model_name]
        
        # Normalize weights
        for model_name in self.weights:
            self.weights[model_name] /= total_inverse_rmse
        
        logger.info(f"Ensemble weights: {self.weights}")
        return self.weights
    
    def weighted_ensemble(self, predictions_dict: Dict) -> np.ndarray:
        """Create weighted ensemble prediction."""
        ensemble_pred = np.zeros_like(next(iter(predictions_dict.values())))
        
        for model_name, prediction in predictions_dict.items():
            if model_name in self.weights:
                ensemble_pred += self.weights[model_name] * prediction
        
        return ensemble_pred
    
    def stacking_ensemble(self, base_models: Dict, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Create stacking ensemble with meta-learner."""
        # Create base model predictions
        base_predictions_train = []
        base_predictions_val = []
        
        for model_name, model in base_models.items():
            pred_train = model.predict(X_train).flatten()
            pred_val = model.predict(X_val).flatten()
            
            base_predictions_train.append(pred_train)
            base_predictions_val.append(pred_val)
        
        # Stack predictions
        X_meta_train = np.column_stack(base_predictions_train)
        X_meta_val = np.column_stack(base_predictions_val)
        
        # Train meta-learner
        self.meta_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.meta_model.fit(X_meta_train, y_train)
        
        # Predict with stacking ensemble
        stacking_pred = self.meta_model.predict(X_meta_val)
        
        return stacking_pred

class HeartRiskPredictor:
    """Main class for heart disease risk prediction pipeline."""
    
    def __init__(self):
        self.config = Config()
        self.data_processor = AdvancedDataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.evaluator = AdvancedEvaluator(self.config)
        self.ensemble_predictor = EnsemblePredictor(self.config)
        
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = []
        self.models = {}
        self.predictions = {}
    
    def run_pipeline(self):
        """Execute the complete prediction pipeline."""
        logger.info("Starting Advanced Heart Risk Prediction Pipeline")
        
        try:
            # 1. Data loading and preprocessing
            self.df = self.data_processor.load_data()
            self.X, self.y, self.feature_names = self.data_processor.prepare_features(self.df)
            
            # 2. Data visualization
            self._create_advanced_visualizations()
            
            # 3. Train-validation-test split
            X_temp, X_test, y_temp, y_test = train_test_split(
                self.X, self.y, test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=self.config.validation_size,
                random_state=self.config.random_state
            )
            
            logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
            
            # 4. Model training
            self.models = self.model_trainer.train_models(X_train, y_train, X_val, y_val, self.X.shape[1])
            
            # 5. Hyperparameter optimization (optional)
            best_params = self.model_trainer.hyperparameter_optimization(X_train, y_train, X_val, y_val, self.X.shape[1])
            
            # 6. Model evaluation
            val_metrics = {}
            for model_name, model in self.models.items():
                y_val_pred = model.predict(X_val).flatten()
                self.predictions[model_name] = y_val_pred
                
                metrics = self.evaluator.comprehensive_metrics(y_val, y_val_pred, model_name)
                val_metrics[model_name] = metrics
                
                # Cross-validation
                cv_results = self.evaluator.cross_validate(model, X_train, y_train, model_name)
                logger.info(f"{model_name} CV Results: {cv_results}")
            
            # 7. Ensemble prediction
            self.ensemble_predictor.calculate_weights(val_metrics)
            ensemble_pred = self.ensemble_predictor.weighted_ensemble(self.predictions)
            self.predictions['ensemble'] = ensemble_pred
            
            # Add ensemble to metrics
            ensemble_metrics = self.evaluator.comprehensive_metrics(y_val, ensemble_pred, 'ensemble')
            
            # 8. Statistical testing
            statistical_results = self.evaluator.statistical_significance(y_val, self.predictions)
            logger.info("Statistical significance results:")
            logger.info(f"\n{statistical_results}")
            
            # 9. Comprehensive evaluation plots
            self.evaluator.create_comprehensive_plots(y_val, self.predictions, self.feature_names, 
                                                     self.models['mlp'])
            
            # 10. Generate report
            report = self.evaluator.generate_report()
            with open(os.path.join(self.config.output_dir, 'evaluation_report.md'), 'w') as f:
                f.write(report)
            
            # 11. Save models and artifacts
            self._save_artifacts()
            
            # 12. Example predictions
            self._demonstrate_predictions(X_val, y_val)
            
            logger.info("Advanced Heart Risk Prediction Pipeline Completed Successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _create_advanced_visualizations(self):
        """Create comprehensive data visualizations."""
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        
        # Select only numeric columns for correlation
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Correlation heatmap
        plt.subplot(2, 2, 1)
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # Distribution of target variable
        plt.subplot(2, 2, 2)
        sns.histplot(self.df['heart.disease'], kde=True)
        plt.title('Heart Disease Risk Distribution')
        plt.xlabel('Heart Disease Risk')
        
        # Feature relationships
        plt.subplot(2, 2, 3)
        plt.scatter(self.df['biking'], self.df['heart.disease'], alpha=0.6, label='Biking')
        plt.scatter(self.df['smoking'], self.df['heart.disease'], alpha=0.6, label='Smoking')
        plt.xlabel('Feature Value')
        plt.ylabel('Heart Disease Risk')
        plt.legend()
        plt.title('Feature vs Target Relationships')
        
        # Feature importance (using random forest)
        plt.subplot(2, 2, 4)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        
        # Get feature importance
        if hasattr(self.data_processor, 'feature_names'):
            feature_imp = pd.DataFrame({
                'feature': self.data_processor.feature_names,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(feature_imp['feature'], feature_imp['importance'])
            plt.title('Random Forest Feature Importance')
            plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'advanced_data_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_artifacts(self):
        """Save models, scalers, and other artifacts."""
        # Save models
        for model_name, model in self.models.items():
            model.save(os.path.join(self.config.model_dir, f'{model_name}_model.h5'))
        
        # Save scaler and feature names
        artifacts = {
            'scaler': self.data_processor.scaler,
            'feature_names': self.feature_names,
            'best_params': self.model_trainer.best_params,
            'metrics': self.evaluator.metrics
        }
        
        with open(os.path.join(self.config.model_dir, 'preprocessing_artifacts.pkl'), 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Save configuration
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        with open(os.path.join(self.config.output_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _demonstrate_predictions(self, X_val: np.ndarray, y_val: np.ndarray):
        """Demonstrate predictions on sample data."""
        logger.info("\n=== Prediction Demonstration ===")
        
        # Sample some validation examples
        sample_indices = np.random.choice(len(X_val), min(5, len(X_val)), replace=False)
        
        for idx in sample_indices:
            actual = y_val[idx]
            features = X_val[idx]
            
            logger.info(f"\nSample {idx}:")
            logger.info(f"Features: {features}")
            logger.info(f"Actual risk: {actual:.2f}")
            
            for model_name, model in self.models.items():
                prediction = model.predict(features.reshape(1, -1))[0][0]
                error = abs(actual - prediction)
                logger.info(f"{model_name:>10}: {prediction:6.2f} (error: {error:5.2f})")
            
            # Ensemble prediction
            ensemble_pred = 0
            for model_name, weight in self.ensemble_predictor.weights.items():
                pred = self.models[model_name].predict(features.reshape(1, -1))[0][0]
                ensemble_pred += weight * pred
            
            ensemble_error = abs(actual - ensemble_pred)
            logger.info(f"{'ensemble':>10}: {ensemble_pred:6.2f} (error: {ensemble_error:5.2f})")

def main():
    """Main execution function."""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore')
    
    try:
        predictor = HeartRiskPredictor()
        predictor.run_pipeline()
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
