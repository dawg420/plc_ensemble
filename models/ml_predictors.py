# models/ml_predictors_fixed.py
"""
Fixed ML Model Predictors (XGBoost, Random Forest, LSTM)
Combined service that can run any of the ML models
Now imports from shared model classes to avoid pickle issues
"""

import pickle
import time
import sys
import threading
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.state_store import StateStore
from shared.utils import ModelTimer, safe_model_prediction, log_ensemble_event
# Import shared model classes to ensure they're available for pickle
from shared.model_classes import (
    XGBoostStatePredictor,
    RandomForestStatePredictor, 
    LSTMStatePredictor
)

class MLPredictorService:
    """Generic ML Model Prediction Service"""
    
    def __init__(self, model_type, model_path, store_path="shared_state.json"):
        self.model_name = model_type.lower()
        self.model_type = model_type
        self.model_path = model_path
        self.store = StateStore(store_path)
        self.model = None
        self.is_running = False
        self.prediction_count = 0
        self.sequence_history = []  # For LSTM
        
        # Load model
        self._load_model()
        
        # Register with state store
        self.store.register_model(self.model_name, "ready")
        
    def _load_model(self):
        """Load the trained ML model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            log_ensemble_event("MODEL_LOADED", f"{self.model_type} model loaded from {self.model_path}")
            return True
        except Exception as e:
            log_ensemble_event("ERROR", f"Failed to load {self.model_type} model: {e}")
            return False
    
    def _convert_enhanced_to_simple_state(self, enhanced_state):
        """Convert enhanced state format to simple format for model"""
        simple_state = {
            "holding_registers": {},
            "coils": {}
        }
        
        # Extract values and reconstruct format expected by the model
        for i in range(39):
            reg_data = enhanced_state["holding_registers"].get(str(i), {
                "value": 0, "last_changed_seconds": 0, "total_changes": 0
            })
            simple_state["holding_registers"][i] = {
                "value": reg_data["value"],
                "last_changed": enhanced_state["metadata"]["sequence_id"] - reg_data["last_changed_seconds"],
                "change_count": reg_data["total_changes"]
            }
        
        for i in range(19):
            coil_data = enhanced_state["coils"].get(str(i), {
                "value": 0, "last_changed_seconds": 0, "total_changes": 0
            })
            simple_state["coils"][i] = {
                "value": coil_data["value"],
                "last_changed": enhanced_state["metadata"]["sequence_id"] - coil_data["last_changed_seconds"],
                "change_count": coil_data["total_changes"]
            }
        
        return simple_state
    
    def _calculate_prediction_confidence(self, current_state, predicted_state, model_type):
        """Calculate confidence score based on model type and prediction"""
        # Count total changes
        total_changes = 0
        total_positions = 39 + 19
        
        # Count changes in holding registers
        for i in range(39):
            current_val = current_state.get("holding_registers", {}).get(i, {}).get("value", 0)
            predicted_val = predicted_state.get("holding_registers", {}).get(i, 0)
            if current_val != predicted_val:
                total_changes += 1
        
        # Count changes in coils
        for i in range(19):
            current_val = current_state.get("coils", {}).get(i, {}).get("value", 0)
            predicted_val = predicted_state.get("coils", {}).get(i, 0)
            if current_val != predicted_val:
                total_changes += 1
        
        change_ratio = total_changes / total_positions
        
        # Different confidence models for different ML approaches
        if model_type.lower() == "xgboost":
            # XGBoost is generally confident, scales with number of changes
            confidence = max(0.7, 0.95 - change_ratio * 0.3)
        elif model_type.lower() == "random_forest":
            # Random Forest is moderate confidence
            confidence = max(0.5, 0.85 - change_ratio * 0.4)
        elif model_type.lower() == "lstm":
            # LSTM confidence depends on sequence length
            sequence_bonus = min(0.2, len(self.sequence_history) / 50.0)
            confidence = max(0.6, 0.8 - change_ratio * 0.3 + sequence_bonus)
        else:
            # Default confidence
            confidence = max(0.5, 0.8 - change_ratio * 0.3)
        
        return confidence
    
    def _update_sequence_history(self, state):
        """Update sequence history for LSTM model"""
        if self.model_type.lower() == "lstm":
            from shared.utils import state_to_features
            
            # Convert state to features
            features = state_to_features(
                state["holding_registers"],
                state["coils"],
                0  # current_time not needed for history
            )
            
            self.sequence_history.append(features)
            
            # Keep only recent history (LSTM sequence length is typically 10)
            max_history = getattr(self.model, 'sequence_length', 10) * 2
            if len(self.sequence_history) > max_history:
                self.sequence_history = self.sequence_history[-max_history:]
    
    def process_prediction_request(self, request_data):
        """Process a single prediction request"""
        request_id = request_data["request_id"]
        current_state = request_data["current_state"]
        remaining_time = request_data["remaining_time"]
        
        if remaining_time <= 0:
            log_ensemble_event("TIMEOUT", f"Request expired before processing", request_id)
            return
        
        log_ensemble_event("PROCESSING", f"{self.model_type} processing request", request_id)
        
        with ModelTimer(self.model_name) as timer:
            try:
                # Convert enhanced state to format expected by model
                simple_state = self._convert_enhanced_to_simple_state(current_state)
                current_time = current_state["metadata"]["sequence_id"]
                
                # Update sequence history for LSTM
                self._update_sequence_history(simple_state)
                
                # Make prediction with appropriate parameters
                if self.model_type.lower() == "lstm" and hasattr(self.model, 'sequence_length'):
                    prediction, error = safe_model_prediction(
                        self.model.predict_next_state,
                        simple_state,
                        current_time=current_time,
                        sequence_history=self.sequence_history
                    )
                else:
                    prediction, error = safe_model_prediction(
                        self.model.predict_next_state,
                        simple_state,
                        current_time=current_time
                    )
                
                if prediction is None:
                    log_ensemble_event("ERROR", f"{self.model_type} prediction failed: {error}", request_id)
                    return
                
                # Calculate confidence
                confidence = self._calculate_prediction_confidence(
                    current_state, prediction, self.model_type
                )
                
                # Submit prediction to state store
                processing_time = time.time() - timer.start_time
                self.store.submit_prediction(
                    request_id=request_id,
                    model_name=self.model_name,
                    predicted_state=prediction,
                    confidence=confidence,
                    processing_time=processing_time
                )
                
                self.prediction_count += 1
                log_ensemble_event("SUCCESS", 
                    f"{self.model_type} prediction submitted (confidence: {confidence:.3f})\n Prediction: {prediction}", 
                    request_id)
                
            except Exception as e:
                log_ensemble_event("ERROR", f"{self.model_type} prediction error: {e}", request_id)
    
    def run_prediction_loop(self, poll_interval=0.5):
        """Main prediction loop"""
        log_ensemble_event("STARTUP", 
            f"{self.model_type} prediction service starting (poll every {poll_interval}s)")
        self.is_running = True
        
        while self.is_running:
            try:
                # Get pending requests
                pending_requests = self.store.get_pending_requests(self.model_name)
                
                if pending_requests:
                    log_ensemble_event("POLLING", f"Found {len(pending_requests)} pending requests")
                    
                    # Process each request
                    for request_data in pending_requests:
                        if not self.is_running:
                            break
                        self.process_prediction_request(request_data)
                
                # Update model status
                self.store.register_model(self.model_name, "active")
                
                # Sleep before next poll
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                log_ensemble_event("SHUTDOWN", f"{self.model_type} service interrupted by user")
                break
            except Exception as e:
                log_ensemble_event("ERROR", f"{self.model_type} service error: {e}")
                time.sleep(poll_interval * 2)  # Longer sleep on error
        
        self.is_running = False
        self.store.register_model(self.model_name, "stopped")
        log_ensemble_event("SHUTDOWN", 
            f"{self.model_type} service stopped. Total predictions: {self.prediction_count}")
    
    def stop(self):
        """Stop the prediction service"""
        self.is_running = False

# Factory functions for each model type
def run_xgboost_service(model_path="models/saved_models/xgboost_model.pkl", 
                       store_path="shared_state.json"):
    """Run XGBoost service"""
    service = MLPredictorService("XGBoost", model_path, store_path)
    service.run_prediction_loop()
    return service

def run_random_forest_service(model_path="models/saved_models/random_forest_model.pkl", 
                             store_path="shared_state.json"):
    """Run Random Forest service"""
    service = MLPredictorService("Random_Forest", model_path, store_path)
    service.run_prediction_loop()
    return service

def run_lstm_service(model_path="models/saved_models/lstm_model.pkl", 
                    store_path="shared_state.json"):
    """Run LSTM service"""
    service = MLPredictorService("LSTM", model_path, store_path)
    service.run_prediction_loop(poll_interval=0.8)  # Slightly slower for LSTM
    return service

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ML Prediction Service")
    parser.add_argument("model_type", choices=["xgboost", "random_forest", "lstm"],
                       help="Type of ML model to run")
    parser.add_argument("--model_path", default=None,
                       help="Path to model file (auto-detected if not provided)")
    parser.add_argument("--store_path", default="shared_state.json",
                       help="Path to shared state file")
    
    args = parser.parse_args()
    
    # Auto-detect model path if not provided
    if args.model_path is None:
        args.model_path = f"models/saved_models/{args.model_type}_model.pkl"
    
    print(f"🤖 Starting {args.model_type.upper()} Prediction Service...")
    print("Press Ctrl+C to stop")
    
    try:
        service = MLPredictorService(args.model_type.capitalize(), args.model_path, args.store_path)
        service.run_prediction_loop()
    except KeyboardInterrupt:
        print(f"\n👋 {args.model_type.upper()} service stopped by user")
    except Exception as e:
        print(f"❌ {args.model_type.upper()} service error: {e}")