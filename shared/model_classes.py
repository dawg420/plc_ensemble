# shared/model_classes_fixed.py
"""
Fixed model classes that only predict VALUES, not temporal metadata
"""

import numpy as np
import copy
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from hmmlearn import hmm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

try:
    from .utils import (
        state_to_features_for_training, 
        state_to_values_only,
        values_to_prediction_state,
        features_to_state  # For training compatibility
    )
except ImportError:
    # Fallback for direct execution
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
        self.value_scaler = RobustScaler()  # Separate scaler for values only
        self.is_fitted = False
        
    def train(self, features, train_split=0.95):
        """Train the HMM using full features for temporal context"""
        print(f"Training HMM on {features.shape[0]} states...")
        
        # Use only training data
        train_size = int(len(features) * train_split)
        train_features = features[:train_size]
        
        # Train on full features (for temporal context)
        features_scaled = self.scaler.fit_transform(train_features)
        self.model.fit(features_scaled)
        
        # Also fit scaler for values-only predictions
        values_only = []
        for feature_row in train_features:
            # Extract values: positions 0, 3, 6, ... for registers, then similar for coils
            values = []
            # Holding registers (39 values)
            for i in range(39):
                values.append(feature_row[i * 3])  # Every 3rd element starting from 0
            # Coils (19 values)  
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
        
        # Convert current state to full features for HMM context
        current_features = state_to_features_for_training(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        # Use HMM to predict next full feature state
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        current_hidden_state = self.model.predict(current_features_scaled)[0]
        
        # Sample next state from HMM
        next_hidden_state = np.random.choice(self.n_components, p=self.model.transmat_[current_hidden_state])
        next_features_scaled = np.random.multivariate_normal(
            self.model.means_[next_hidden_state],
            self.model.covars_[next_hidden_state]
        )
        
        next_features = self.scaler.inverse_transform(next_features_scaled.reshape(1, -1))[0]
        
        # Extract only the VALUES from predicted features (ignore predicted temporal data)
        predicted_values = []
        # Holding registers values (every 3rd element starting from 0)
        for i in range(39):
            predicted_values.append(next_features[i * 3])
        # Coils values (every 3rd element starting from coils section)
        coils_start = 39 * 3
        for i in range(19):
            predicted_values.append(next_features[coils_start + i * 3])
        
        # Convert values to proper state with temporal metadata
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)

class LSTMStatePredictor:
    """LSTM Model - predicts values only"""
    
    def __init__(self, sequence_length=10, lstm_units=128, dropout_rate=0.3):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dim = 174  # Full features for training
        self.value_dim = 58     # Values only for prediction (39 + 19)
        
    def build_model(self):
        """Build LSTM model for full features"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(256, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(128, activation='relu'),
            Dense(self.value_dim, activation='linear')  # Output only values
        ])
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def create_sequences(self, features):
        """Create input-output sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            
            # Extract only values from target (not temporal metadata)
            target_features = features[i]
            target_values = []
            # Holding registers values
            for j in range(39):
                target_values.append(target_features[j * 3])
            # Coils values
            coils_start = 39 * 3
            for j in range(19):
                target_values.append(target_features[coils_start + j * 3])
            
            y.append(target_values)
        
        return np.array(X), np.array(y)
    
    def train(self, features, train_split=0.95):
        """Train LSTM"""
        print(f"Training LSTM on {features.shape[0]} states...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled)
        
        # Train/validation split
        train_size = int(len(X) * train_split)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss')
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print(f"LSTM training complete. Best val_loss: {min(history.history['val_loss']):.6f}")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0, sequence_history=None):
        """Predict next state using LSTM - values only"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        # Build sequence from history or current state
        if sequence_history is None or len(sequence_history) < self.sequence_length:
            # Use current state repeated
            current_features = state_to_features_for_training(
                current_state_dict["holding_registers"], 
                current_state_dict["coils"],
                current_time
            )
            sequence = np.tile(current_features, (self.sequence_length, 1))
        else:
            sequence = np.array(sequence_history[-self.sequence_length:])
        
        # Scale and predict
        sequence_scaled = self.scaler.transform(sequence)
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        # Model outputs values only
        predicted_values_scaled = self.model.predict(sequence_scaled, verbose=0)[0]
        
        # Inverse transform - need to be careful here since we're only predicting values
        # Create a dummy full feature vector for inverse transform
        dummy_features = np.zeros(self.feature_dim)
        # Place predicted values in appropriate positions
        for i in range(39):  # Holding registers
            dummy_features[i * 3] = predicted_values_scaled[i]
        coils_start = 39 * 3
        for i in range(19):  # Coils
            dummy_features[coils_start + i * 3] = predicted_values_scaled[39 + i]
        
        # Inverse transform the dummy vector
        dummy_features_unscaled = self.scaler.inverse_transform(dummy_features.reshape(1, -1))[0]
        
        # Extract the unscaled values
        predicted_values = []
        for i in range(39):
            predicted_values.append(dummy_features_unscaled[i * 3])
        for i in range(19):
            predicted_values.append(dummy_features_unscaled[coils_start + i * 3])
        
        # Convert values to proper state with temporal metadata
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)

class XGBoostStatePredictor:
    """XGBoost Model - predicts values only"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_dim = 174  # Full features for input
        self.value_dim = 58     # Values only for output
        
    def train(self, features, train_split=0.95):
        """Train XGBoost to predict values only"""
        print(f"Training XGBoost on {features.shape[0]} states...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create X (full features), y (values only)
        X = features_scaled[:-1]
        
        # Extract values from next state
        y_values = []
        for i in range(1, len(features_scaled)):
            values = []
            # Holding registers values
            for j in range(39):
                values.append(features_scaled[i][j * 3])
            # Coils values
            coils_start = 39 * 3
            for j in range(19):
                values.append(features_scaled[i][coils_start + j * 3])
            y_values.append(values)
        
        y = np.array(y_values)
        
        # Train/test split
        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        # XGBoost parameters
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
            'tree_method': 'hist'
        }
        
        # Train single model to predict all values
        self.model = xgb.XGBRegressor(**xgb_params)
        self.model.fit(X_train, y_train, verbose=False)
        
        self.is_fitted = True
        print("XGBoost training complete!")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0):
        """Predict next state using XGBoost - values only"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        # Convert current state to full features
        current_features = state_to_features_for_training(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        
        # Predict values only
        predicted_values_scaled = self.model.predict(current_features_scaled)[0]
        
        # Inverse transform values (create dummy feature vector)
        dummy_features = np.zeros(self.feature_dim)
        for i in range(39):  # Holding registers
            dummy_features[i * 3] = predicted_values_scaled[i]
        coils_start = 39 * 3
        for i in range(19):  # Coils
            dummy_features[coils_start + i * 3] = predicted_values_scaled[39 + i]
        
        dummy_features_unscaled = self.scaler.inverse_transform(dummy_features.reshape(1, -1))[0]
        
        # Extract unscaled values
        predicted_values = []
        for i in range(39):
            predicted_values.append(dummy_features_unscaled[i * 3])
        for i in range(19):
            predicted_values.append(dummy_features_unscaled[coils_start + i * 3])
        
        # Convert values to proper state with temporal metadata
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)

class RandomForestStatePredictor:
    """Random Forest Model - predicts values only"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_dim = 174
        self.value_dim = 58
        
    def train(self, features, train_split=0.95):
        """Train Random Forest to predict values only"""
        print(f"Training Random Forest on {features.shape[0]} states...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create X (full features), y (values only)
        X = features_scaled[:-1]
        
        # Extract values from next state
        y_values = []
        for i in range(1, len(features_scaled)):
            values = []
            # Holding registers values
            for j in range(39):
                values.append(features_scaled[i][j * 3])
            # Coils values
            coils_start = 39 * 3
            for j in range(19):
                values.append(features_scaled[i][coils_start + j * 3])
            y_values.append(values)
        
        y = np.array(y_values)
        
        # Train/test split
        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        # Random Forest parameters
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
        """Predict next state using Random Forest - values only"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        # Convert current state to full features
        current_features = state_to_features_for_training(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        predicted_values_scaled = self.model.predict(current_features_scaled)[0]
        
        # Inverse transform values
        dummy_features = np.zeros(self.feature_dim)
        for i in range(39):  # Holding registers
            dummy_features[i * 3] = predicted_values_scaled[i]
        coils_start = 39 * 3
        for i in range(19):  # Coils
            dummy_features[coils_start + i * 3] = predicted_values_scaled[39 + i]
        
        dummy_features_unscaled = self.scaler.inverse_transform(dummy_features.reshape(1, -1))[0]
        
        # Extract unscaled values
        predicted_values = []
        for i in range(39):
            predicted_values.append(dummy_features_unscaled[i * 3])
        for i in range(19):
            predicted_values.append(dummy_features_unscaled[coils_start + i * 3])
        
        # Convert values to proper state with temporal metadata
        return values_to_prediction_state(predicted_values, current_state_dict, current_time)