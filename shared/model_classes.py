# shared/model_classes_fixed_gpu.py
"""
Fixed model classes with proper GPU memory management for TensorFlow
"""

import numpy as np
import copy
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from hmmlearn import hmm

# Configure TensorFlow BEFORE importing it
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# CRITICAL: Configure GPU memory growth to prevent TensorFlow from hogging all GPU memory
def configure_tensorflow_gpu():
    """Configure TensorFlow to use only the GPU memory it needs"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Alternative: Set explicit memory limit if needed
            # tf.config.experimental.set_memory_limit(gpus[0], 4096)  # 4GB limit
            
            print(f"✅ Configured TensorFlow GPU memory growth for {len(gpus)} GPU(s)")
            
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("ℹ️  No GPUs detected by TensorFlow")

# Configure GPU before any TensorFlow operations
configure_tensorflow_gpu()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

try:
    from .utils import (
        state_to_features_for_training, 
        state_to_values_only,
        values_to_prediction_state,
        features_to_state
    )
except ImportError:
    import sys
    sys.path.append('.')
    from shared.utils import (
        state_to_features_for_training,
        state_to_values_only, 
        values_to_prediction_state,
        features_to_state
    )

warnings.filterwarnings('ignore')

class HMMStatePredictor:
    """HMM Model - predicts values only, not temporal metadata"""
    
    def __init__(self, n_components=25):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            tol=1e-3,
            init_params="kmeans",
            algorithm="viterbi",
            random_state=42,
            verbose=False
        )
        self.scaler = RobustScaler()
        self.value_scaler = RobustScaler()
        self.is_fitted = False
        
    def train(self, features, train_split=0.95):
        """Train the HMM using full features for temporal context"""
        print(f"Training HMM on {features.shape[0]} states...")
        
        train_size = int(len(features) * train_split)
        train_features = features[:train_size]
        
        features_scaled = self.scaler.fit_transform(train_features)
        self.model.fit(features_scaled)
        
        values_only = []
        for feature_row in train_features:
            values = []
            for i in range(39):
                values.append(feature_row[i * 3])
            coils_start = 39 * 3
            for i in range(19):
                values.append(feature_row[coils_start + i * 3])
            values_only.append(values)
        
        values_only = np.array(values_only)
        self.value_scaler.fit(values_only)
        
        self.is_fitted = True
        print(f"HMM training complete. Converged: {self.model.monitor_.converged}")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0):
        """Predict next state - only values, then compute temporal metadata"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        current_features = state_to_features_for_training(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        current_hidden_state = self.model.predict(current_features_scaled)[0]
        
        next_hidden_state = np.random.choice(self.n_components, p=self.model.transmat_[current_hidden_state])
        next_features_scaled = np.random.multivariate_normal(
            self.model.means_[next_hidden_state],
            self.model.covars_[next_hidden_state]
        )
        
        next_features = self.scaler.inverse_transform(next_features_scaled.reshape(1, -1))[0]
        
        predicted_values = []
        for i in range(39):
            predicted_values.append(next_features[i * 3])
        coils_start = 39 * 3
        for i in range(19):
            predicted_values.append(next_features[coils_start + i * 3])
        
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)

class LSTMStatePredictor:
    """LSTM Model with proper GPU memory management"""
    
    def __init__(self, sequence_length=10, lstm_units=128, dropout_rate=0.3):
        print("🔧 Initializing LSTM with GPU memory management...")
        
        # Ensure TensorFlow GPU is properly configured
        configure_tensorflow_gpu()
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dim = 174
        self.value_dim = 58
        
        # Check GPU availability and memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ LSTM will use GPU: {gpus[0]}")
            # Check current memory usage
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"   Current GPU memory: {gpu_info['current'] / 1024**3:.1f}GB / {gpu_info['peak'] / 1024**3:.1f}GB peak")
            except:
                print("   GPU memory info not available")
        else:
            print("⚠️  LSTM will use CPU (no GPU detected)")
        
    def build_model(self):
        """Build LSTM model with explicit GPU placement"""
        print("🏗️  Building LSTM model architecture...")
        
        # Build model with explicit device placement if GPU available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            with tf.device('/GPU:0'):
                model = Sequential([
                    LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
                    Dropout(self.dropout_rate),
                    LSTM(self.lstm_units // 2, return_sequences=False),
                    Dropout(self.dropout_rate),
                    Dense(256, activation='relu'),
                    Dropout(self.dropout_rate),
                    Dense(128, activation='relu'),
                    Dense(self.value_dim, activation='linear')
                ])
        else:
            model = Sequential([
                LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units // 2, return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(256, activation='relu'),
                Dropout(self.dropout_rate),
                Dense(128, activation='relu'),
                Dense(self.value_dim, activation='linear')
            ])
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Print model memory usage
        total_params = model.count_params()
        print(f"   LSTM model parameters: {total_params:,}")
        print(f"   Estimated model size: ~{total_params * 4 / 1024**2:.1f}MB")
        
        return model
    
    def create_sequences(self, features):
        """Create input-output sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            
            target_features = features[i]
            target_values = []
            for j in range(39):
                target_values.append(target_features[j * 3])
            coils_start = 39 * 3
            for j in range(19):
                target_values.append(target_features[coils_start + j * 3])
            
            y.append(target_values)
        
        return np.array(X), np.array(y)
    
    def train(self, features, train_split=0.95):
        """Train LSTM with memory monitoring"""
        print(f"Training LSTM on {features.shape[0]} states...")
        print(f"🔍 Monitoring GPU memory during training...")
        
        # Check initial GPU memory
        try:
            if tf.config.experimental.list_physical_devices('GPU'):
                initial_memory = tf.config.experimental.get_memory_info('GPU:0')
                print(f"   Initial GPU memory: {initial_memory['current'] / 1024**3:.1f}GB")
        except:
            pass
        
        features_scaled = self.scaler.fit_transform(features)
        X, y = self.create_sequences(features_scaled)
        
        train_size = int(len(X) * train_split)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        self.model = self.build_model()
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss')
        ]
        
        # Train with smaller batch size to be memory-efficient
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,  # Smaller batch size to conserve memory
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Check final GPU memory
        try:
            if tf.config.experimental.list_physical_devices('GPU'):
                final_memory = tf.config.experimental.get_memory_info('GPU:0')
                print(f"   Final GPU memory: {final_memory['current'] / 1024**3:.1f}GB")
                print(f"   Peak GPU memory: {final_memory['peak'] / 1024**3:.1f}GB")
        except:
            pass
        
        self.is_fitted = True
        print(f"LSTM training complete. Best val_loss: {min(history.history['val_loss']):.6f}")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0, sequence_history=None):
        """Predict next state using LSTM"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        if sequence_history is None or len(sequence_history) < self.sequence_length:
            current_features = state_to_features_for_training(
                current_state_dict["holding_registers"], 
                current_state_dict["coils"],
                current_time
            )
            sequence = np.tile(current_features, (self.sequence_length, 1))
        else:
            sequence = np.array(sequence_history[-self.sequence_length:])
        
        sequence_scaled = self.scaler.transform(sequence)
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        predicted_values_scaled = self.model.predict(sequence_scaled, verbose=0)[0]
        
        dummy_features = np.zeros(self.feature_dim)
        for i in range(39):
            dummy_features[i * 3] = predicted_values_scaled[i]
        coils_start = 39 * 3
        for i in range(19):
            dummy_features[coils_start + i * 3] = predicted_values_scaled[39 + i]
        
        dummy_features_unscaled = self.scaler.inverse_transform(dummy_features.reshape(1, -1))[0]
        
        predicted_values = []
        for i in range(39):
            predicted_values.append(dummy_features_unscaled[i * 3])
        for i in range(19):
            predicted_values.append(dummy_features_unscaled[coils_start + i * 3])
        
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)

class XGBoostStatePredictor:
    """XGBoost Model - CPU only (to save GPU memory for LLM)"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_dim = 174
        self.value_dim = 58
        
    def train(self, features, train_split=0.95):
        """Train XGBoost to predict values only"""
        print(f"Training XGBoost on {features.shape[0]} states (CPU-only)...")
        
        features_scaled = self.scaler.fit_transform(features)
        
        X = features_scaled[:-1]
        
        y_values = []
        for i in range(1, len(features_scaled)):
            values = []
            for j in range(39):
                values.append(features_scaled[i][j * 3])
            coils_start = 39 * 3
            for j in range(19):
                values.append(features_scaled[i][coils_start + j * 3])
            y_values.append(values)
        
        y = np.array(y_values)
        
        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'  # CPU only - no GPU
        }
        
        self.model = xgb.XGBRegressor(**xgb_params)
        self.model.fit(X_train, y_train, verbose=False)
        
        self.is_fitted = True
        print("XGBoost training complete!")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0):
        """Predict next state using XGBoost"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        current_features = state_to_features_for_training(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        predicted_values_scaled = self.model.predict(current_features_scaled)[0]
        
        dummy_features = np.zeros(self.feature_dim)
        for i in range(39):
            dummy_features[i * 3] = predicted_values_scaled[i]
        coils_start = 39 * 3
        for i in range(19):
            dummy_features[coils_start + i * 3] = predicted_values_scaled[39 + i]
        
        dummy_features_unscaled = self.scaler.inverse_transform(dummy_features.reshape(1, -1))[0]
        
        predicted_values = []
        for i in range(39):
            predicted_values.append(dummy_features_unscaled[i * 3])
        for i in range(19):
            predicted_values.append(dummy_features_unscaled[coils_start + i * 3])
        
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)

class RandomForestStatePredictor:
    """Random Forest Model - CPU only"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_dim = 174
        self.value_dim = 58
        
    def train(self, features, train_split=0.95):
        """Train Random Forest to predict values only"""
        print(f"Training Random Forest on {features.shape[0]} states (CPU-only)...")
        
        features_scaled = self.scaler.fit_transform(features)
        
        X = features_scaled[:-1]
        
        y_values = []
        for i in range(1, len(features_scaled)):
            values = []
            for j in range(39):
                values.append(features_scaled[i][j * 3])
            coils_start = 39 * 3
            for j in range(19):
                values.append(features_scaled[i][coils_start + j * 3])
            y_values.append(values)
        
        y = np.array(y_values)
        
        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = RandomForestRegressor(**rf_params)
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        print("Random Forest training complete!")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0):
        """Predict next state using Random Forest"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        current_features = state_to_features_for_training(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        predicted_values_scaled = self.model.predict(current_features_scaled)[0]
        
        dummy_features = np.zeros(self.feature_dim)
        for i in range(39):
            dummy_features[i * 3] = predicted_values_scaled[i]
        coils_start = 39 * 3
        for i in range(19):
            dummy_features[coils_start + i * 3] = predicted_values_scaled[39 + i]
        
        dummy_features_unscaled = self.scaler.inverse_transform(dummy_features.reshape(1, -1))[0]
        
        predicted_values = []
        for i in range(39):
            predicted_values.append(dummy_features_unscaled[i * 3])
        for i in range(19):
            predicted_values.append(dummy_features_unscaled[coils_start + i * 3])
        
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)