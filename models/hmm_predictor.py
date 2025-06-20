# models/hmm_predictor_fixed.py
"""
Fixed HMM Model Predictor
Monitors shared state store and provides HMM predictions
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

from shared.distributed_state import DistributedStateStore
from shared.utils import ModelTimer, safe_model_prediction, log_ensemble_event
# Import the shared model class to ensure it's available for pickle
from shared.model_classes import HMMStatePredictor

class HMMPredictorService:
    """HMM Model Prediction Service"""
    
    def __init__(self, model_path="models/saved_models/hmm_model.pkl", 
                 base_dir="."):
        self.model_name = "hmm"
        self.model_path = model_path
        self.store = DistributedStateStore(base_dir)
        self.model = None
        self.is_running = False
        self.prediction_count = 0
        
        # Load model
        self._load_model()
        
        # Register with state store
        self.store.register_model(self.model_name, "ready")
        
    def _load_model(self):
        """Load the trained HMM model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            log_ensemble_event("MODEL_LOADED", f"HMM model loaded from {self.model_path}")
            return True
        except Exception as e:
            log_ensemble_event("ERROR", f"Failed to load HMM model: {e}")
            return False
    
    def _convert_enhanced_to_simple_state(self, enhanced_state):
        """Convert enhanced state format to simple format for model"""
        simple_state = {
            "holding_registers": {},
            "coils": {}
        }
        
        # Extract just the values and reconstruct the format expected by the model
        for i in range(39):
            reg_data = enhanced_state["holding_registers"].get(str(i), {"value": 0, "last_changed_seconds": 0, "total_changes": 0})
            simple_state["holding_registers"][i] = {
                "value": reg_data["value"],
                "last_changed": enhanced_state["metadata"]["sequence_id"] - reg_data["last_changed_seconds"],
                "change_count": reg_data["total_changes"]
            }
        
        for i in range(19):
            coil_data = enhanced_state["coils"].get(str(i), {"value": 0, "last_changed_seconds": 0, "total_changes": 0})
            simple_state["coils"][i] = {
                "value": coil_data["value"],
                "last_changed": enhanced_state["metadata"]["sequence_id"] - coil_data["last_changed_seconds"],
                "change_count": coil_data["total_changes"]
            }
        
        return simple_state
    
    def _calculate_prediction_confidence(self, current_state, predicted_state):
        """Calculate confidence score for the prediction"""
        # Simple confidence based on how much the state changes
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
        
        # Confidence decreases with more changes (HMM tends to be conservative)
        change_ratio = total_changes / total_positions
        confidence = max(0.3, 1.0 - change_ratio * 2.0)  # Scale confidence
        
        return confidence
    
    def process_prediction_request(self, request_data):
        """Process a single prediction request - SIMPLIFIED"""
        request_id = request_data["request_id"]
        current_state = request_data["current_state"]
        remaining_time = request_data.get("remaining_time", 0)
        
        if remaining_time <= 0:
            log_ensemble_event("TIMEOUT", f"Request expired before processing", request_id)
            return
        
        log_ensemble_event("PROCESSING", f"HMM processing request from individual queue", request_id)
        
        with ModelTimer(self.model_name) as timer:
            try:
                # Convert enhanced state to format expected by model
                simple_state = self._convert_enhanced_to_simple_state(current_state)
                current_time = current_state["metadata"]["sequence_id"]
                
                # Make prediction
                prediction, error = safe_model_prediction(
                    self.model.predict_next_state,
                    simple_state,
                    current_time=current_time
                )
                
                if prediction is None:
                    log_ensemble_event("ERROR", f"HMM prediction failed: {error}", request_id)
                    return
                
                # Calculate confidence
                confidence = self._calculate_prediction_confidence(current_state, prediction)
                
                # MODIFIED: Submit prediction to shared response file
                processing_time = time.time() - timer.start_time
                self.store.submit_prediction(
                    request_id=request_id,
                    model_name=self.model_name,
                    predicted_state=prediction,
                    confidence=confidence,
                    processing_time=processing_time
                )
                
                self.prediction_count += 1
                log_ensemble_event("SUCCESS", f"HMM prediction submitted to shared responses (confidence: {confidence:.3f})", request_id)
                
            except Exception as e:
                log_ensemble_event("ERROR", f"HMM prediction error: {e}", request_id)
    
    def run_prediction_loop(self, poll_interval=0.5):
        """Main prediction loop"""
        log_ensemble_event("STARTUP", f"HMM prediction service starting (poll every {poll_interval}s)")
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
                log_ensemble_event("SHUTDOWN", "HMM service interrupted by user")
                break
            except Exception as e:
                log_ensemble_event("ERROR", f"HMM service error: {e}")
                time.sleep(poll_interval * 2)  # Longer sleep on error
        
        self.is_running = False
        self.store.register_model(self.model_name, "stopped")
        log_ensemble_event("SHUTDOWN", f"HMM service stopped. Total predictions: {self.prediction_count}")
    
    def stop(self):
        """Stop the prediction service"""
        self.is_running = False

def run_hmm_service_thread(model_path="models/saved_models/hmm_model.pkl", 
                          base_dir="."):
    """Run HMM service in a separate thread"""
    service = HMMPredictorService(model_path, base_dir)
    service.run_prediction_loop()
    return service

if __name__ == "__main__":
    # Run as standalone service
    print("🧠 Starting HMM Prediction Service...")
    print("Press Ctrl+C to stop")
    
    try:
        service = HMMPredictorService()
        service.run_prediction_loop()
    except KeyboardInterrupt:
        print("\n👋 HMM service stopped by user")
    except Exception as e:
        print(f"❌ HMM service error: {e}")