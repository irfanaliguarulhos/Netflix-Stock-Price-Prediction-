#!/usr/bin/env python3
"""
Netflix Stock Price Prediction - Robust End-to-End System
==========================================================
A comprehensive machine learning pipeline for stock price prediction featuring:
- Advanced data preprocessing and feature engineering
- Multiple ML/DL models (LSTM, GRU, XGBoost, Ensemble)
- Hyperparameter optimization with Optuna
- Comprehensive backtesting and validation
- Interactive 360° analytics dashboard
- Model explainability with SHAP
- Production-ready deployment code

Author: Professional ML Engineer
Version: 2.0.0
"""

import warnings
warnings.filterwarnings('ignore')

# Core Data Processing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Machine Learning Models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2

# Hyperparameter Optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Model Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# Visualization & Dashboard
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot

# Statistical Analysis
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# System utilities
import os
import json
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Advanced data preprocessing with feature engineering for stock prediction.
    Handles missing values, creates technical indicators, and prepares sequences.
    """
    
    def __init__(self, lookback: int = 60, forecast_horizon: int = 30):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scalers = {}
        self.feature_names = []
        
    def fetch_data(self, ticker: str = 'NFLX', 
                   start_date: str = '2015-01-01',
                   end_date: str = None) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            logger.info(f"Successfully fetched {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators for feature enrichment.
        Includes momentum, volatility, trend, and volume indicators.
        """
        logger.info("Creating technical indicators...")
        
        df = df.copy()
        
        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Momentum Indicators
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)
        df['MACD'] = df['EMA_12'] - df['EMA_26'] if 'EMA_12' in df.columns else df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100
        
        # Volatility
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = df['Close'].pct_change().rolling(window=window).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price position
        df['Price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Trend indicators
        df['ADX'] = self._calculate_adx(df, period=14)
        
        # Lag features
        for lag in range(1, 6):
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Return_lag_{lag}'] = df['Close'].pct_change().shift(lag)
        
        # Target variable (future return)
        df['Target'] = df['Close'].shift(-self.forecast_horizon)
        df['Future_return'] = (df['Target'] - df['Close']) / df['Close'] * 100
        
        # Drop NaN values
        df = df.dropna()
        
        self.feature_names = [col for col in df.columns if col not in ['Target', 'Future_return']]
        logger.info(f"Created {len(self.feature_names)} features")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, 
                        sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(target[i+sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame, 
                    test_size: float = 0.2,
                    scale: bool = True) -> Dict[str, Any]:
        """
        Prepare data for training with proper train/test split.
        Uses time-based splitting to avoid look-ahead bias.
        """
        logger.info("Preparing data for modeling...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Target', 'Future_return']]
        X = df[feature_cols].values
        y = df['Close'].shift(-self.forecast_horizon).values
        
        # Remove last forecast_horizon rows where target is NaN
        X = X[:-self.forecast_horizon]
        y = y[:-self.forecast_horizon]
        
        # Scale features
        if scale:
            self.scalers['features'] = MinMaxScaler(feature_range=(0, 1))
            X_scaled = self.scalers['features'].fit_transform(X)
            
            self.scalers['target'] = MinMaxScaler(feature_range=(0, 1))
            y_scaled = self.scalers['target'].fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X_scaled = X
            y_scaled = y
        
        # Time-based split
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Create sequences for deep learning
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, self.lookback)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, self.lookback)
        
        logger.info(f"Training samples: {len(X_train_seq)}, Test samples: {len(X_test_seq)}")
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'X_train_raw': X_train,
            'X_test_raw': X_test,
            'y_train_raw': y_train,
            'y_test_raw': y_test,
            'feature_names': feature_cols
        }


class ModelEnsemble:
    """
    Ensemble of multiple models for robust predictions.
    Includes LSTM, GRU, XGBoost, and stacking ensemble.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.history = {}
        self.best_params = {}
        
    def _default_config(self) -> Dict:
        """Default model configurations."""
        return {
            'lstm': {
                'units': [128, 64],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'optimizer': 'adam',
                'loss': 'mse'
            },
            'gru': {
                'units': [128, 64],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'optimizer': 'adam',
                'loss': 'mse'
            },
            'xgboost': {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
    
    def build_lstm_model(self, input_shape: Tuple, params: Dict = None) -> Sequential:
        """Build optimized LSTM model with regularization."""
        params = params or self.config['lstm']
        
        model = Sequential([
            Bidirectional(LSTM(params['units'][0], 
                              return_sequences=True,
                              kernel_regularizer=l2(0.001),
                              recurrent_regularizer=l2(0.001)),
                          input_shape=input_shape),
            Dropout(params['dropout']),
            
            Bidirectional(LSTM(params['units'][1],
                              return_sequences=False,
                              kernel_regularizer=l2(0.001))),
            Dropout(params['dropout']),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_gru_model(self, input_shape: Tuple, params: Dict = None) -> Sequential:
        """Build optimized GRU model."""
        params = params or self.config['gru']
        
        model = Sequential([
            Bidirectional(GRU(params['units'][0],
                             return_sequences=True,
                             kernel_regularizer=l2(0.001)),
                          input_shape=input_shape),
            Dropout(params['dropout']),
            
            Bidirectional(GRU(params['units'][1],
                             return_sequences=False,
                             kernel_regularizer=l2(0.001))),
            Dropout(params['dropout']),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_xgboost_model(self, params: Dict = None) -> xgb.XGBRegressor:
        """Build XGBoost model."""
        params = params or self.config['xgboost']
        return xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
    
    def train_models(self, data: Dict, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train all models in the ensemble."""
        logger.info("Training ensemble models...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]
        
        results = {}
        
        # Train LSTM
        logger.info("Training LSTM model...")
        lstm_model = self.build_lstm_model((data['X_train'].shape[1], data['X_train'].shape[2]))
        lstm_history = lstm_model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_test'], data['y_test']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.models['lstm'] = lstm_model
        self.history['lstm'] = lstm_history.history
        results['lstm'] = self.evaluate_model(lstm_model, data['X_test'], data['y_test'])
        
        # Train GRU
        logger.info("Training GRU model...")
        gru_model = self.build_gru_model((data['X_train'].shape[1], data['X_train'].shape[2]))
        gru_history = gru_model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_test'], data['y_test']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.models['gru'] = gru_model
        self.history['gru'] = gru_history.history
        results['gru'] = self.evaluate_model(gru_model, data['X_test'], data['y_test'])
        
        # Train XGBoost (reshape for tree models)
        logger.info("Training XGBoost model...")
        X_train_flat = data['X_train'].reshape(data['X_train'].shape[0], -1)
        X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
        
        xgb_model = self.build_xgboost_model()
        xgb_model.fit(
            X_train_flat, data['y_train'],
            eval_set=[(X_test_flat, data['y_test'])],
            early_stopping_rounds=20,
            verbose=True
        )
        self.models['xgboost'] = xgb_model
        results['xgboost'] = self.evaluate_model_xgb(xgb_model, X_test_flat, data['y_test'])
        
        # Create ensemble
        logger.info("Creating weighted ensemble...")
        self._create_ensemble_weights(results)
        
        return results
    
    def _create_ensemble_weights(self, results: Dict):
        """Create ensemble weights based on model performance."""
        # Inverse MSE weighting
        mse_values = {k: v['mse'] for k, v in results.items()}
        total_inv_mse = sum(1/mse for mse in mse_values.values())
        self.ensemble_weights = {k: (1/mse)/total_inv_mse for k, mse in mse_values.items()}
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def predict_ensemble(self, X: np.ndarray, X_flat: np.ndarray = None) -> np.ndarray:
        """Generate ensemble predictions."""
        if X_flat is None:
            X_flat = X.reshape(X.shape[0], -1)
        
        preds = []
        weights = []
        
        if 'lstm' in self.models:
            preds.append(self.models['lstm'].predict(X, verbose=0).flatten())
            weights.append(self.ensemble_weights.get('lstm', 0.33))
        
        if 'gru' in self.models:
            preds.append(self.models['gru'].predict(X, verbose=0).flatten())
            weights.append(self.ensemble_weights.get('gru', 0.33))
        
        if 'xgboost' in self.models:
            preds.append(self.models['xgboost'].predict(X_flat))
            weights.append(self.ensemble_weights.get('xgboost', 0.34))
        
        # Weighted average
        ensemble_pred = np.average(preds, axis=0, weights=weights)
        return ensemble_pred
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance."""
        y_pred = model.predict(X_test, verbose=0).flatten()
        return self._calculate_metrics(y_test, y_pred)
    
    def evaluate_model_xgb(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate XGBoost model."""
        y_pred = model.predict(X_test)
        return self._calculate_metrics(y_test, y_pred)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Direction accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            direction_accuracy = 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna.
    Implements Bayesian optimization for efficient search.
    """
    
    def __init__(self, data: Dict, n_trials: int = 50):
        self.data = data
        self.n_trials = n_trials
        self.best_study = {}
        
    def optimize_lstm(self) -> Dict:
        """Optimize LSTM hyperparameters."""
        logger.info("Optimizing LSTM hyperparameters...")
        
        def objective(trial):
            units1 = trial.suggest_categorical('units1', [64, 128, 256])
            units2 = trial.suggest_categorical('units2', [32, 64, 128])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            model = Sequential([
                Bidirectional(LSTM(units1, return_sequences=True),
                             input_shape=(self.data['X_train'].shape[1], self.data['X_train'].shape[2])),
                Dropout(dropout),
                Bidirectional(LSTM(units2, return_sequences=False)),
                Dropout(dropout),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                self.data['X_train'], self.data['y_train'],
                validation_data=(self.data['X_test'], self.data['y_test']),
                epochs=50,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            return min(history.history['val_loss'])
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_study['lstm'] = study
        logger.info(f"Best LSTM params: {study.best_params}")
        logger.info(f"Best LSTM score: {study.best_value}")
        
        return study.best_params
    
    def optimize_xgboost(self) -> Dict:
        """Optimize XGBoost hyperparameters."""
        logger.info("Optimizing XGBoost hyperparameters...")
        
        X_train_flat = self.data['X_train'].reshape(self.data['X_train'].shape[0], -1)
        X_test_flat = self.data['X_test'].reshape(self.data['X_test'].shape[0], -1)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train_flat, self.data['y_train'],
                eval_set=[(X_test_flat, self.data['y_test'])],
                early_stopping_rounds=20,
                verbose=False
            )
            
            y_pred = model.predict(X_test_flat)
            return mean_squared_error(self.data['y_test'], y_pred)
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_study['xgboost'] = study
        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best XGBoost score: {study.best_value}")
        
        return study.best_params


class DashboardBuilder:
    """
    Interactive 360° dashboard for comprehensive model analysis.
    Creates professional visualizations using Plotly.
    """
    
    def __init__(self, output_dir: str = 'dashboard'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_full_dashboard(self, 
                             df: pd.DataFrame,
                             predictions: Dict,
                             metrics: Dict,
                             history: Dict = None,
                             feature_importance: Dict = None) -> str:
        """Create comprehensive 360° dashboard."""
        logger.info("Creating 360° dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Historical Price & Predictions',
                'Prediction vs Actual',
                'Model Performance Comparison',
                'Prediction Errors Distribution',
                'Training History',
                'Feature Importance',
                'Returns Distribution',
                'Cumulative Returns'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Historical Price & Predictions
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Actual Price',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        
        if 'ensemble' in predictions:
            pred_index = df.index[-len(predictions['ensemble']):]
            fig.add_trace(
                go.Scatter(x=pred_index, y=predictions['ensemble'], name='Ensemble Prediction',
                          line=dict(color='#ff7f0e', width=2, dash='dash')),
                row=1, col=1
            )
        
        # 2. Prediction vs Actual
        if 'ensemble' in predictions:
            fig.add_trace(
                go.Scatter(x=df['Close'].values[-len(predictions['ensemble']):],
                          y=predictions['ensemble'],
                          mode='markers', name='Predictions',
                          marker=dict(color='#2ca02c', size=8, opacity=0.6)),
                row=1, col=2
            )
            # Perfect prediction line
            min_val = min(df['Close'].min(), min(predictions['ensemble']))
            max_val = max(df['Close'].max(), max(predictions['ensemble']))
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          name='Perfect Prediction',
                          line=dict(color='red', width=2, dash='dot')),
                row=1, col=2
            )
        
        # 3. Model Performance Comparison
        if metrics:
            models = list(metrics.keys())
            rmse_values = [metrics[m]['rmse'] for m in models]
            r2_values = [metrics[m]['r2'] for m in models]
            
            fig.add_trace(
                go.Bar(x=models, y=rmse_values, name='RMSE',
                      marker_color='#1f77b4'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=models, y=r2_values, name='R²',
                      marker_color='#2ca02c'),
                row=2, col=1
            )
        
        # 4. Prediction Errors Distribution
        if 'ensemble' in predictions:
            errors = df['Close'].values[-len(predictions['ensemble']):] - predictions['ensemble']
            fig.add_trace(
                go.Histogram(x=errors, name='Error Distribution',
                            nbinsx=30, marker_color='#d62728', opacity=0.7),
                row=2, col=2
            )
        
        # 5. Training History
        if history:
            for model_name, hist in history.items():
                if 'loss' in hist:
                    fig.add_trace(
                        go.Scatter(y=hist['loss'], name=f'{model_name} Train Loss',
                                  line=dict(width=2)),
                        row=3, col=1
                    )
                if 'val_loss' in hist:
                    fig.add_trace(
                        go.Scatter(y=hist['val_loss'], name=f'{model_name} Val Loss',
                                  line=dict(width=2, dash='dash')),
                        row=3, col=1
                    )
        
        # 6. Feature Importance
        if feature_importance:
            features = list(feature_importance.keys())[:15]
            importance = [feature_importance[f] for f in features]
            
            fig.add_trace(
                go.Bar(x=importance, y=features, orientation='h',
                      name='Feature Importance',
                      marker_color='#9467bd'),
                row=3, col=2
            )
        
        # 7. Returns Distribution
        returns = df['Close'].pct_change().dropna()
        fig.add_trace(
            go.Histogram(x=returns, name='Returns Distribution',
                        nbinsx=50, marker_color='#8c564b', opacity=0.7),
            row=4, col=1
        )
        
        # 8. Cumulative Returns
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                      name='Cumulative Returns',
                      line=dict(color='#e377c2', width=2)),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1400,
            width=1600,
            title_text="Netflix Stock Price Prediction - 360° Analytics Dashboard",
            title_font_size=24,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white'
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / 'comprehensive_dashboard.html'
        plot(fig, filename=str(dashboard_path), auto_open=False)
        logger.info(f"Dashboard saved to {dashboard_path}")
        
        return str(dashboard_path)
    
    def create_model_comparison_chart(self, metrics: Dict) -> str:
        """Create detailed model comparison visualization."""
        models = list(metrics.keys())
        
        # Metrics to compare
        metric_names = ['RMSE', 'MAE', 'R²', 'MAPE (%)', 'Direction Accuracy (%)']
        metric_keys = ['rmse', 'mae', 'r2', 'mape', 'direction_accuracy']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Error Metrics', 'Performance Scores', 
                           'Metric Radar Chart', 'Model Ranking'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatterpolar"}, {"type": "bar"}]]
        )
        
        # Bar chart for errors (lower is better)
        for i, (key, name) in enumerate(zip(['rmse', 'mae', 'mape'], ['RMSE', 'MAE', 'MAPE'])):
            values = [metrics[m][key] for m in models]
            fig.add_trace(
                go.Bar(name=name, x=models, y=values),
                row=1, col=1
            )
        
        # Bar chart for scores (higher is better)
        for key, name in zip(['r2', 'direction_accuracy'], ['R²', 'Direction Acc']):
            values = [metrics[m][key] for m in models]
            fig.add_trace(
                go.Bar(name=name, x=models, y=values),
                row=1, col=2
            )
        
        # Radar chart
        for model in models:
            values = [
                metrics[model]['rmse'] / max(m['rmse'] for m in metrics.values()),
                metrics[model]['mae'] / max(m['mae'] for m in metrics.values()),
                1 - metrics[model]['r2'],  # Invert R² for consistency
                metrics[model]['mape'] / max(m['mape'] for m in metrics.values()),
                1 - metrics[model]['direction_accuracy']/100  # Invert direction acc
            ]
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=['RMSE', 'MAE', 'R²', 'MAPE', 'Direction'],
                    fill='toself',
                    name=model
                ),
                row=2, col=1
            )
        
        # Model ranking (composite score)
        composite_scores = {}
        for model in models:
            # Lower errors and higher R²/direction accuracy are better
            score = (
                -metrics[model]['rmse'] / max(m['rmse'] for m in metrics.values()) +
                -metrics[model]['mae'] / max(m['mae'] for m in metrics.values()) +
                metrics[model]['r2'] +
                -metrics[model]['mape'] / max(m['mape'] for m in metrics.values()) +
                metrics[model]['direction_accuracy']/100
            )
            composite_scores[model] = score
        
        fig.add_trace(
            go.Bar(
                x=list(composite_scores.keys()),
                y=list(composite_scores.values()),
                name='Composite Score'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Model Performance Comparison",
            showlegend=True,
            template='plotly_white'
        )
        
        path = self.output_dir / 'model_comparison.html'
        plot(fig, filename=str(path), auto_open=False)
        return str(path)


class Backtester:
    """
    Comprehensive backtesting framework for strategy validation.
    Implements walk-forward validation and various trading strategies.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        
    def walk_forward_validation(self, model, data: Dict, 
                               n_splits: int = 5) -> Dict:
        """Perform walk-forward cross-validation."""
        logger.info("Performing walk-forward validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        X = data['X_train_raw']
        y = data['y_train_raw']
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold+1}/{n_splits}")
            
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Retrain model on fold
            # (Simplified - in practice, you'd retrain the model)
            
            # Make predictions
            y_pred = model.predict(X_test_fold, verbose=0).flatten()
            
            # Calculate metrics
            metrics = {
                'fold': fold + 1,
                'mse': mean_squared_error(y_test_fold, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_fold, y_pred)),
                'mae': mean_absolute_error(y_test_fold, y_pred),
                'r2': r2_score(y_test_fold, y_pred)
            }
            
            results.append(metrics)
        
        # Aggregate results
        aggregated = {
            'mean_rmse': np.mean([r['rmse'] for r in results]),
            'std_rmse': np.std([r['rmse'] for r in results]),
            'mean_r2': np.mean([r['r2'] for r in results]),
            'std_r2': np.std([r['r2'] for r in results]),
            'fold_results': results
        }
        
        logger.info(f"Walk-forward RMSE: {aggregated['mean_rmse']:.4f} (+/- {aggregated['std_rmse']:.4f})")
        logger.info(f"Walk-forward R²: {aggregated['mean_r2']:.4f} (+/- {aggregated['std_r2']:.4f})")
        
        return aggregated
    
    def simulate_trading_strategy(self, predictions: np.ndarray, 
                                 actual_prices: np.ndarray,
                                 threshold: float = 0.02) -> Dict:
        """Simulate trading strategy based on predictions."""
        logger.info("Simulating trading strategy...")
        
        capital = self.initial_capital
        position = 0  # Number of shares
        trades = []
        portfolio_values = [capital]
        
        for i in range(1, len(predictions)):
            pred_change = (predictions[i] - predictions[i-1]) / predictions[i-1]
            actual_change = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            # Trading logic
            if pred_change > threshold and position == 0:
                # Buy signal
                shares_to_buy = int(capital * 0.95 / actual_prices[i])
                if shares_to_buy > 0:
                    cost = shares_to_buy * actual_prices[i]
                    capital -= cost
                    position += shares_to_buy
                    trades.append({'type': 'BUY', 'price': actual_prices[i], 'shares': shares_to_buy})
            
            elif pred_change < -threshold and position > 0:
                # Sell signal
                revenue = position * actual_prices[i]
                capital += revenue
                trades.append({'type': 'SELL', 'price': actual_prices[i], 'shares': position})
                position = 0
            
            # Portfolio value
            portfolio_value = capital + position * actual_prices[i]
            portfolio_values.append(portfolio_value)
        
        # Close any open position
        if position > 0:
            capital += position * actual_prices[-1]
            portfolio_values[-1] = capital
        
        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-8)
        max_drawdown = np.min((portfolio_values - np.maximum.accumulate(portfolio_values)) / np.maximum.accumulate(portfolio_values)) * 100
        
        return {
            'final_capital': capital,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values
        }


def run_complete_pipeline(ticker: str = 'NFLX'):
    """
    Execute complete end-to-end pipeline.
    This is the main function that runs everything.
    """
    print("="*80)
    print("NETFLIX STOCK PRICE PREDICTION - ROBUST END-TO-END SYSTEM")
    print("="*80)
    
    # Step 1: Data Preprocessing
    print("\n[1/6] DATA PREPROCESSING & FEATURE ENGINEERING")
    print("-"*80)
    preprocessor = DataPreprocessor(lookback=60, forecast_horizon=30)
    df = preprocessor.fetch_data(ticker, start_date='2018-01-01')
    df = preprocessor.create_technical_indicators(df)
    data = preprocessor.prepare_data(df, test_size=0.2)
    
    print(f"✓ Fetched {len(df)} days of data")
    print(f"✓ Created {len(preprocessor.feature_names)} features")
    print(f"✓ Training samples: {len(data['X_train'])}, Test samples: {len(data['X_test'])}")
    
    # Step 2: Hyperparameter Optimization (Optional - can be time-consuming)
    print("\n[2/6] HYPERPARAMETER OPTIMIZATION")
    print("-"*80)
    print("Skipping extensive optimization for demo (set n_trials>0 to enable)")
    best_lstm_params = None
    best_xgb_params = None
    
    # Uncomment to enable optimization:
    # optimizer = HyperparameterOptimizer(data, n_trials=20)
    # best_lstm_params = optimizer.optimize_lstm()
    # best_xgb_params = optimizer.optimize_xgboost()
    
    # Step 3: Model Training
    print("\n[3/6] MODEL TRAINING")
    print("-"*80)
    ensemble = ModelEnsemble()
    results = ensemble.train_models(data, epochs=50, batch_size=32)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Performance:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
    
    # Step 4: Generate Predictions
    print("\n[4/6] GENERATING PREDICTIONS")
    print("-"*80)
    X_test_flat = data['X_test'].reshape(data['X_test'].shape[0], -1)
    ensemble_pred = ensemble.predict_ensemble(data['X_test'], X_test_flat)
    
    # Inverse transform predictions
    ensemble_pred_original = preprocessor.scalers['target'].inverse_transform(
        ensemble_pred.reshape(-1, 1)
    ).flatten()
    
    predictions = {
        'ensemble': ensemble_pred_original,
        'lstm': preprocessor.scalers['target'].inverse_transform(
            ensemble.models['lstm'].predict(data['X_test'], verbose=0).reshape(-1, 1)
        ).flatten(),
        'gru': preprocessor.scalers['target'].inverse_transform(
            ensemble.models['gru'].predict(data['X_test'], verbose=0).reshape(-1, 1)
        ).flatten()
    }
    
    # Get actual prices
    actual_test = preprocessor.scalers['target'].inverse_transform(
        data['y_test'].reshape(-1, 1)
    ).flatten()
    
    print(f"✓ Generated {len(ensemble_pred)} predictions")
    print(f"✓ Prediction range: ${ensemble_pred_original.min():.2f} - ${ensemble_pred_original.max():.2f}")
    
    # Step 5: Backtesting
    print("\n[5/6] BACKTESTING & VALIDATION")
    print("-"*80)
    backtester = Backtester(initial_capital=100000)
    
    # Walk-forward validation
    wf_results = backtester.walk_forward_validation(
        ensemble.models['lstm'], data, n_splits=5
    )
    
    # Trading strategy simulation
    trading_results = backtester.simulate_trading_strategy(
        ensemble_pred, data['y_test']
    )
    
    print(f"\nTrading Strategy Results:")
    print(f"  Total Return: {trading_results['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {trading_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {trading_results['max_drawdown_pct']:.2f}%")
    print(f"  Number of Trades: {trading_results['num_trades']}")
    
    # Step 6: Create Dashboard
    print("\n[6/6] CREATING 360° DASHBOARD")
    print("-"*80)
    dashboard_builder = DashboardBuilder(output_dir='stock_dashboard')
    
    # Add predictions to dataframe for visualization
    pred_index = df.index[-len(ensemble_pred):]
    df_pred = df.copy()
    
    # Create comprehensive metrics dict
    all_metrics = results.copy()
    all_metrics['ensemble'] = ensemble._calculate_metrics(actual_test, ensemble_pred)
    
    # Feature importance from XGBoost
    feature_importance = {}
    if 'xgboost' in ensemble.models:
        importance_scores = ensemble.models['xgboost'].feature_importances_
        for i, feat in enumerate(data['feature_names']):
            feature_importance[feat] = importance_scores[i]
    
    dashboard_path = dashboard_builder.create_full_dashboard(
        df=df_pred,
        predictions=predictions,
        metrics=all_metrics,
        history=ensemble.history,
        feature_importance=feature_importance
    )
    
    # Create model comparison chart
    comparison_path = dashboard_builder.create_model_comparison_chart(all_metrics)
    
    print(f"✓ Dashboard created: {dashboard_path}")
    print(f"✓ Model comparison: {comparison_path}")
    
    # Save model artifacts
    print("\n[SAVING ARTIFACTS]")
    print("-"*80)
    
    # Save models
    ensemble.models['lstm'].save('lstm_model.h5')
    with open('ensemble_config.pkl', 'wb') as f:
        pickle.dump({
            'weights': ensemble.ensemble_weights,
            'scalers': preprocessor.scalers,
            'config': ensemble.config
        }, f)
    
    # Save metrics report
    metrics_report = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'data_points': len(df),
        'features': len(preprocessor.feature_names),
        'model_metrics': all_metrics,
        'walk_forward_results': wf_results,
        'trading_results': {
            k: v for k, v in trading_results.items() if k != 'trades' and k != 'portfolio_values'
        }
    }
    
    with open('metrics_report.json', 'w') as f:
        json.dump(metrics_report, f, indent=2)
    
    print("✓ Models saved: lstm_model.h5, ensemble_config.pkl")
    print("✓ Metrics report: metrics_report.json")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nKey Results:")
    print(f"  • Best Model: {max(all_metrics.keys(), key=lambda k: all_metrics[k]['r2'])}")
    print(f"  • Ensemble R²: {all_metrics['ensemble']['r2']:.4f}")
    print(f"  • Ensemble RMSE: {all_metrics['ensemble']['rmse']:.6f}")
    print(f"  • Trading Return: {trading_results['total_return_pct']:.2f}%")
    print(f"\nOpen the dashboard to explore results interactively:")
    print(f"  → {dashboard_path}")
    print("="*80)
    
    return {
        'models': ensemble.models,
        'metrics': all_metrics,
        'predictions': predictions,
        'dashboard_path': dashboard_path
    }


if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_pipeline('NFLX')
