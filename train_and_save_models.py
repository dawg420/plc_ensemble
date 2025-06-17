# train_and_save_models.py
"""
Script to train and save all ML models for the ensemble system
Run this once to prepare all models before starting the ensemble
"""

import pandas as pd
import numpy as np
import pickle
import copy
import warnings
from pathlib import Path
import time

# ML libraries
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from hmmlearn import hmm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Local imports
from shared.utils import update_state_with_transaction, state_to_features, features_to_state

warnings.filterwarnings('ignore')

# ===== BASE CLASSES FOR EACH MODEL =====

class HMMStatePredictor:
    """HMM Model with temporal features"""
    
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
        self.is_fitted = False
        
    def process_dataset(self, file_path, max_rows=20000):
        """Process dataset to extract state sequences"""
        print("Processing dataset for HMM...")
        df = pd.read_csv(file_path)
        
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        holding_registers = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)}
        coils = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
        
        state_sequence = []
        
        for idx, row in df.iterrows():
            current_state_features = state_to_features(holding_registers, coils, idx)
            state_sequence.append(current_state_features)
            update_state_with_transaction(row, idx, holding_registers, coils)
            
            if idx % 2000 == 0 and idx > 0:
                print(f"  Processed {idx} transactions...")
        
        final_state_features = state_to_features(holding_registers, coils, len(df))
        state_sequence.append(final_state_features)
        
        return np.array(state_sequence)
    
    def train(self, features, train_split=0.95):
        """Train the HMM"""
        print(f"Training HMM on {features.shape[0]} states...")
        
        # Use only training data
        train_size = int(len(features) * train_split)
        train_features = features[:train_size]
        
        features_scaled = self.scaler.fit_transform(train_features)
        self.model.fit(features_scaled)
        self.is_fitted = True
        print(f"HMM training complete. Converged: {self.model.monitor_.converged}")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0):
        """Predict next state"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
            
        current_features = state_to_features(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        current_hidden_state = self.model.predict(current_features_scaled)[0]
        
        # Sample next state
        next_hidden_state = np.random.choice(self.n_components, p=self.model.transmat_[current_hidden_state])
        next_features_scaled = np.random.multivariate_normal(
            self.model.means_[next_hidden_state],
            self.model.covars_[next_hidden_state]
        )
        
        next_features = self.scaler.inverse_transform(next_features_scaled.reshape(1, -1))[0]
        next_holding_registers, next_coils = features_to_state(next_features)
        
        return {"holding_registers": next_holding_registers, "coils": next_coils}

class LSTMStatePredictor:
    """LSTM Neural Network Model"""
    
    def __init__(self, sequence_length=10, lstm_units=128, dropout_rate=0.3):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dim = 174  # (39+19)*3
        
    def build_model(self):
        """Build LSTM model"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(256, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(128, activation='relu'),
            Dense(self.feature_dim, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def process_dataset(self, file_path, max_rows=20000):
        """Process dataset for LSTM"""
        print("Processing dataset for LSTM...")
        df = pd.read_csv(file_path)
        
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        holding_registers = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)}
        coils = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
        
        state_sequence = []
        
        for idx, row in df.iterrows():
            current_state_features = state_to_features(holding_registers, coils, idx)
            state_sequence.append(current_state_features)
            update_state_with_transaction(row, idx, holding_registers, coils)
            
            if idx % 2000 == 0 and idx > 0:
                print(f"  Processed {idx} transactions...")
        
        final_state_features = state_to_features(holding_registers, coils, len(df))
        state_sequence.append(final_state_features)
        
        return np.array(state_sequence)
    
    def create_sequences(self, features):
        """Create input-output sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(features[i])
        
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
        """Predict next state using LSTM"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        current_features = state_to_features(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        # Use sequence history if available
        if sequence_history is None or len(sequence_history) < self.sequence_length:
            sequence = np.tile(current_features, (self.sequence_length, 1))
        else:
            sequence = np.array(sequence_history[-self.sequence_length:])
        
        # Scale and predict
        sequence_scaled = self.scaler.transform(sequence)
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, self.feature_dim)
        
        prediction_scaled = self.model.predict(sequence_scaled, verbose=0)[0]
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(1, -1))[0]
        
        next_holding_registers, next_coils = features_to_state(prediction)
        
        return {"holding_registers": next_holding_registers, "coils": next_coils}

class XGBoostStatePredictor:
    """XGBoost Model"""
    
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_dim = 174
        
    def process_dataset(self, file_path, max_rows=20000):
        """Process dataset for XGBoost"""
        print("Processing dataset for XGBoost...")
        df = pd.read_csv(file_path)
        
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        holding_registers = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)}
        coils = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
        
        state_sequence = []
        
        for idx, row in df.iterrows():
            current_state_features = state_to_features(holding_registers, coils, idx)
            state_sequence.append(current_state_features)
            update_state_with_transaction(row, idx, holding_registers, coils)
            
            if idx % 2000 == 0 and idx > 0:
                print(f"  Processed {idx} transactions...")
        
        final_state_features = state_to_features(holding_registers, coils, len(df))
        state_sequence.append(final_state_features)
        
        return np.array(state_sequence)
    
    def train(self, features, train_split=0.95):
        """Train XGBoost models"""
        print(f"Training XGBoost on {features.shape[0]} states...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create X, y pairs
        X = features_scaled[:-1]
        y = features_scaled[1:]
        
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
        
        # Train separate models for different feature groups
        feature_groups = {
            'holding_reg_values': list(range(0, 39*3, 3)),
            'holding_reg_temporal': list(range(1, 39*3, 3)) + list(range(2, 39*3, 3)),
            'coil_values': list(range(39*3, 39*3 + 19*3, 3)),
            'coil_temporal': list(range(39*3 + 1, 39*3 + 19*3, 3)) + list(range(39*3 + 2, 39*3 + 19*3, 3))
        }
        
        self.models = {}
        
        for group_name, feature_indices in feature_groups.items():
            print(f"  Training {group_name} model...")
            
            model = xgb.XGBRegressor(**xgb_params)
            y_group = y_train[:, feature_indices]
            model.fit(X_train, y_group, verbose=False)
            
            self.models[group_name] = {
                'model': model,
                'feature_indices': feature_indices
            }
        
        self.is_fitted = True
        print("XGBoost training complete!")
        return self
    
    def predict_next_state(self, current_state_dict, current_time=0):
        """Predict next state using XGBoost"""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        current_features = state_to_features(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        
        # Predict using each model and combine results
        prediction_scaled = np.zeros(self.feature_dim)
        
        for group_name, model_info in self.models.items():
            model = model_info['model']
            feature_indices = model_info['feature_indices']
            
            group_prediction = model.predict(current_features_scaled)[0]
            prediction_scaled[feature_indices] = group_prediction
        
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(1, -1))[0]
        next_holding_registers, next_coils = features_to_state(prediction)
        
        return {"holding_registers": next_holding_registers, "coils": next_coils}

class RandomForestStatePredictor:
    """Random Forest Model"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_dim = 174
        
    def process_dataset(self, file_path, max_rows=20000):
        """Process dataset for Random Forest"""
        print("Processing dataset for Random Forest...")
        df = pd.read_csv(file_path)
        
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        holding_registers = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)}
        coils = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
        
        state_sequence = []
        
        for idx, row in df.iterrows():
            current_state_features = state_to_features(holding_registers, coils, idx)
            state_sequence.append(current_state_features)
            update_state_with_transaction(row, idx, holding_registers, coils)
            
            if idx % 2000 == 0 and idx > 0:
                print(f"  Processed {idx} transactions...")
        
        final_state_features = state_to_features(holding_registers, coils, len(df))
        state_sequence.append(final_state_features)
        
        return np.array(state_sequence)
    
    def train(self, features, train_split=0.95):
        """Train Random Forest"""
        print(f"Training Random Forest on {features.shape[0]} states...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create X, y pairs
        X = features_scaled[:-1]
        y = features_scaled[1:]
        
        # Train/test split
        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        # Optimized Random Forest parameters
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
        
        current_features = state_to_features(
            current_state_dict["holding_registers"], 
            current_state_dict["coils"],
            current_time
        )
        
        current_features_scaled = self.scaler.transform(np.array(current_features).reshape(1, -1))
        prediction_scaled = self.model.predict(current_features_scaled)[0]
        
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(1, -1))[0]
        next_holding_registers, next_coils = features_to_state(prediction)
        
        return {"holding_registers": next_holding_registers, "coils": next_coils}

# ===== MAIN TRAINING FUNCTION =====

def train_and_save_all_models(dataset_path="modbus_output_with_time.csv", 
                              save_dir="models/saved_models",
                              max_rows=20000,
                              train_split=0.95):
    """Train and save all models"""
    
    print("🚀 Starting Model Training Pipeline")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print(f"Max rows: {max_rows}")
    print(f"Train split: {train_split}")
    print(f"Save directory: {save_dir}")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    models_to_train = {
        # "hmm": HMMStatePredictor(n_components=25),
        "lstm": LSTMStatePredictor(sequence_length=10, lstm_units=128),
        "xgboost": XGBoostStatePredictor(),
        "random_forest": RandomForestStatePredictor()
    }
    
    trained_models = {}
    training_times = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n{'='*20} Training {model_name.upper()} {'='*20}")
        start_time = time.time()
        
        try:
            # Process dataset
            features = model.process_dataset(dataset_path, max_rows)
            print(f"Feature shape: {features.shape}")
            
            # Train model
            trained_model = model.train(features, train_split)
            
            # Save model
            model_path = Path(save_dir) / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(trained_model, f)
            
            training_time = time.time() - start_time
            training_times[model_name] = training_time
            trained_models[model_name] = trained_model
            
            print(f"✅ {model_name.upper()} training complete in {training_time:.2f}s")
            print(f"💾 Saved to: {model_path}")
            
        except Exception as e:
            print(f"❌ {model_name.upper()} training failed: {e}")
            continue
    
    # Save training summary
    summary = {
        "training_times": training_times,
        "dataset_path": dataset_path,
        "max_rows": max_rows,
        "train_split": train_split,
        "feature_dim": 174,
        "models_trained": list(trained_models.keys()),
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = Path(save_dir) / "training_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print("🎉 MODEL TRAINING COMPLETE!")
    print(f"📋 Models trained: {', '.join(trained_models.keys())}")
    print(f"⏱️  Total time: {sum(training_times.values()):.2f}s")
    print(f"📁 All models saved in: {save_dir}")
    print(f"📄 Training summary: {summary_path}")
    
    return trained_models, summary

if __name__ == "__main__":
    # Run the training pipeline
    trained_models, summary = train_and_save_all_models()
    
    print("\n🔍 Quick model test...")
    
    # Test each model with a simple prediction
    test_state = {
        "holding_registers": {i: 0 for i in range(39)},
        "coils": {i: 0 for i in range(19)}
    }
    test_state["holding_registers"][0] = 100
    test_state["coils"][0] = 1
    
    for model_name, model in trained_models.items():
        try:
            prediction = model.predict_next_state(test_state, current_time=1000)
            non_zero_regs = sum(1 for v in prediction["holding_registers"].values() if v != 0)
            non_zero_coils = sum(1 for v in prediction["coils"].values() if v != 0)
            print(f"✅ {model_name}: {non_zero_regs} active registers, {non_zero_coils} active coils")
        except Exception as e:
            print(f"❌ {model_name} test failed: {e}")
    
    print("\n✨ Ready to start ensemble system!")
    print("Next step: Run individual model notebooks and orchestrator")