# shared/state_store.py
import json
import time
import uuid
from pathlib import Path
import threading
from typing import Dict, Optional, Any
import copy

class StateStore:
    """Shared state store for multi-notebook communication"""
    
    def __init__(self, store_path="shared_state.json"):
        self.store_path = Path(store_path)
        self.lock = threading.Lock()
        self._initialize_store()
        
    def _initialize_store(self):
        """Initialize the shared state file"""
        if not self.store_path.exists():
            initial_state = {
                "active_requests": {},
                "model_status": {},
                "performance_history": []
            }
            self._write_store(initial_state)
    
    def _read_store(self) -> Dict:
        """Thread-safe read from store"""
        with self.lock:
            try:
                with open(self.store_path, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {"active_requests": {}, "model_status": {}, "performance_history": []}
    
    def _write_store(self, data: Dict):
        """Thread-safe write to store"""
        with self.lock:
            with open(self.store_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def create_enhanced_state(self, holding_registers: Dict, coils: Dict, current_time: int = 0) -> Dict:
        """
        FIXED: Create enhanced state with proper temporal information
        Now uses real seconds for temporal tracking
        """
        enhanced_state = {
            "holding_registers": {},
            "coils": {},
            "metadata": {
                "timestamp": time.time(),
                "sequence_id": current_time  # Keep for backward compatibility
            }
        }
        
        # Enhanced holding registers with temporal info
        for i in range(39):
            reg_data = holding_registers.get(i, {"value": 0, "last_changed": 0, "change_count": 0})
            enhanced_state["holding_registers"][str(i)] = {
                "value": reg_data["value"],
                "last_changed_seconds": reg_data.get("last_changed", 0),  # FIXED: Use actual seconds
                "total_changes": reg_data.get("change_count", 0)
            }
        
        # Enhanced coils with temporal info  
        for i in range(19):
            coil_data = coils.get(i, {"value": 0, "last_changed": 0, "change_count": 0})
            enhanced_state["coils"][str(i)] = {
                "value": coil_data["value"], 
                "last_changed_seconds": coil_data.get("last_changed", 0),  # FIXED: Use actual seconds
                "total_changes": coil_data.get("change_count", 0)
            }
            
        return enhanced_state
    
    def publish_prediction_request(self, current_state: Dict, timeout: float = 3.0) -> str:
        """Publish a new prediction request"""
        request_id = str(uuid.uuid4())
        
        store_data = self._read_store()
        store_data["active_requests"][request_id] = {
            "current_state": current_state,
            "timeout": timeout,
            "created_at": time.time(),
            "predictions": {},
            "consensus": None,
            "status": "pending"
        }
        self._write_store(store_data)
        
        print(f"📤 Published prediction request: {request_id}")
        return request_id
    
    def submit_prediction(self, request_id: str, model_name: str, predicted_state: Dict, 
                         confidence: float = 1.0, processing_time: float = 0.0):
        """Submit a model's prediction"""
        store_data = self._read_store()
        
        if request_id in store_data["active_requests"]:
            store_data["active_requests"][request_id]["predictions"][model_name] = {
                "predicted_state": predicted_state,
                "confidence": confidence,
                "processing_time": processing_time,
                "submitted_at": time.time()
            }
            self._write_store(store_data)
            print(f"✅ {model_name} submitted prediction for {request_id}")
        else:
            print(f"❌ Request {request_id} not found or expired")
    
    def get_pending_requests(self, model_name: str) -> list:
        """Get pending requests for a specific model"""
        store_data = self._read_store()
        pending = []
        
        current_time = time.time()
        for req_id, req_data in store_data["active_requests"].items():
            # Check if request is still valid and model hasn't responded
            if (req_data["status"] == "pending" and 
                model_name not in req_data["predictions"] and
                (current_time - req_data["created_at"]) < req_data["timeout"]):
                pending.append({
                    "request_id": req_id,
                    "current_state": req_data["current_state"],
                    "timeout": req_data["timeout"],
                    "remaining_time": req_data["timeout"] - (current_time - req_data["created_at"])
                })
        
        return pending
    
    def get_predictions(self, request_id: str) -> Optional[Dict]:
        """Get all predictions for a request"""
        store_data = self._read_store()
        
        if request_id in store_data["active_requests"]:
            return store_data["active_requests"][request_id]["predictions"]
        return None
    
    def mark_request_complete(self, request_id: str, consensus_result: Dict):
        """Mark a request as complete with consensus result"""
        store_data = self._read_store()
        
        if request_id in store_data["active_requests"]:
            store_data["active_requests"][request_id]["consensus"] = consensus_result
            store_data["active_requests"][request_id]["status"] = "complete"
            self._write_store(store_data)
    
    def cleanup_expired_requests(self, max_age: float = 300.0):
        """Clean up old requests"""
        store_data = self._read_store()
        current_time = time.time()
        
        expired_requests = [
            req_id for req_id, req_data in store_data["active_requests"].items()
            if (current_time - req_data["created_at"]) > max_age
        ]
        
        for req_id in expired_requests:
            del store_data["active_requests"][req_id]
        
        if expired_requests:
            self._write_store(store_data)
            print(f"🧹 Cleaned up {len(expired_requests)} expired requests")
    
    def register_model(self, model_name: str, status: str = "ready"):
        """Register a model as active"""
        store_data = self._read_store()
        store_data["model_status"][model_name] = {
            "status": status,
            "last_seen": time.time()
        }
        self._write_store(store_data)
        if status == "ready":
            print(f"📋 Registered {model_name} as {status}")
    
    def get_active_models(self) -> list:
        """Get list of currently active models"""
        store_data = self._read_store()
        current_time = time.time()
        
        active_models = []
        for model_name, model_data in store_data["model_status"].items():
            # Consider model active if seen within last 60 seconds
            if (current_time - model_data["last_seen"]) < 60:
                active_models.append(model_name)
        
        return active_models

# Utility functions for state conversion
def enhanced_to_display_state(enhanced_state: Dict) -> Dict:
    """Convert enhanced state to display format (no temporal info)"""
    display_state = {
        "holding_registers": {},
        "coils": {}
    }
    
    for i in range(39):
        display_state["holding_registers"][i] = enhanced_state["holding_registers"][i]["value"]
    
    for i in range(19):
        display_state["coils"][i] = enhanced_state["coils"][i]["value"]
    
    return display_state

def generate_state_output_summary(holding_registers: Dict, coils: Dict) -> str:
    """Generate display summary (same as your existing function)"""
    summary = "STATE SUMMARY\n"
    
    summary += "Holding Registers:\n"
    summary += "Reg | Value\n"
    for i in range(39):
        value = holding_registers.get(i, 0)
        summary += f"HR{i:02d} | {value:4d}\n"
    
    summary += "Coils:\n"
    summary += "Coil | Value\n"
    for i in range(19):
        value = coils.get(i, 0)
        summary += f"C{i:02d}  | {value:4d}\n"
    
    return summary