# shared/distributed_state.py
"""
Distributed State Manager - Replaces StateStore
Manages individual model queues + shared response collection
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List
import copy

# Import the queue manager and utilities
from .queue_manager import QueueManager

class DistributedStateStore:
    """
    Distributed state store using individual queues + shared responses
    Drop-in replacement for StateStore with improved performance
    """
    
    def __init__(self, base_dir=".", model_names=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Default model names
        if model_names is None:
            model_names = ["llm", "xgboost", "hmm", "lstm", "random_forest"]
        
        self.model_names = model_names
        
        # Initialize queue manager for individual queues
        self.queue_manager = QueueManager(self.base_dir, model_names)
        
        # Shared response and coordination files
        self.responses_file = self.base_dir / "shared_responses.json"
        self.coordination_file = self.base_dir / "coordination.json"
        
        # Locks for shared files
        self.responses_lock = threading.Lock()
        self.coordination_lock = threading.Lock()
        
        # Initialize shared files
        self._initialize_shared_files()
    
    def _initialize_shared_files(self):
        """Initialize shared response and coordination files"""
        # Initialize responses file
        if not self.responses_file.exists():
            with self.responses_lock:
                initial_responses = {
                    "responses": {},
                    "metadata": {
                        "created_at": time.time(),
                        "total_responses": 0
                    }
                }
                self._write_json_file(self.responses_file, initial_responses)
        
        # Initialize coordination file
        if not self.coordination_file.exists():
            with self.coordination_lock:
                initial_coordination = {
                    "model_health": {
                        model: {"status": "unknown", "last_seen": 0}
                        for model in self.model_names
                    },
                    "active_requests": [],
                    "completed_requests": [],
                    "system_metadata": {
                        "created_at": time.time(),
                        "last_cleanup": time.time()
                    }
                }
                self._write_json_file(self.coordination_file, initial_coordination)
    
    def _write_json_file(self, file_path: Path, data: Dict):
        """Atomic write to JSON file"""
        import tempfile
        import os
        
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix=f'{file_path.stem}_',
            dir=file_path.parent
        )
        temp_file = Path(temp_path)
        
        try:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.rename(file_path)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _read_json_file(self, file_path: Path) -> Dict:
        """Read JSON file with error handling"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file_path}: {e}")
            return {}
    
    # ===== REQUEST DISTRIBUTION (Orchestrator Interface) =====
    
    def publish_prediction_request(self, current_state: Dict, timeout: float = 20.0) -> str:
        """
        Publish prediction request to all model queues
        
        Args:
            current_state: Enhanced state format
            timeout: Request timeout in seconds
            
        Returns:
            request_id: Unique identifier for the request
        """
        # Distribute to all model queues atomically
        request_id = self.queue_manager.distribute_request_atomically(current_state, timeout)
        
        # Track in coordination file
        with self.coordination_lock:
            coord_data = self._read_json_file(self.coordination_file)
            coord_data["active_requests"].append({
                "request_id": request_id,
                "created_at": time.time(),
                "timeout": timeout,
                "expected_models": self.model_names.copy()
            })
            self._write_json_file(self.coordination_file, coord_data)
        
        print(f"📤 Published request {request_id[:8]} to {len(self.model_names)} model queues")
        return request_id
    
    def get_predictions(self, request_id: str) -> Optional[Dict]:
        """
        Get all predictions for a request from shared response file
        
        Args:
            request_id: Request identifier
            
        Returns:
            Dictionary of predictions by model name, or None if no predictions
        """
        with self.responses_lock:
            responses_data = self._read_json_file(self.responses_file)
            
            request_responses = responses_data.get("responses", {}).get(request_id)
            if request_responses:
                # Return in format compatible with old StateStore
                return request_responses
            else:
                return None
    
    def mark_request_complete(self, request_id: str, consensus_result: Dict):
        """
        Mark request as complete and eligible for cleanup
        
        Args:
            request_id: Request identifier
            consensus_result: Final consensus result
        """
        with self.coordination_lock:
            coord_data = self._read_json_file(self.coordination_file)
            
            # Move from active to completed
            active_requests = coord_data.get("active_requests", [])
            completed_requests = coord_data.get("completed_requests", [])
            
            # Find and move the request
            for i, req in enumerate(active_requests):
                if req.get("request_id") == request_id:
                    completed_req = active_requests.pop(i)
                    completed_req["completed_at"] = time.time()
                    completed_req["consensus_result"] = consensus_result
                    completed_requests.append(completed_req)
                    break
            
            coord_data["active_requests"] = active_requests
            coord_data["completed_requests"] = completed_requests
            
            self._write_json_file(self.coordination_file, coord_data)
        
        print(f"✅ Marked request {request_id[:8]} as complete")
    
    # ===== MODEL QUEUE INTERFACE (Model Services) =====
    
    def get_pending_requests(self, model_name: str) -> List[Dict]:
        """
        Get pending requests for a specific model (simplified interface)
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of pending request dictionaries
        """
        return self.queue_manager.get_pending_requests(model_name)
    
    def submit_prediction(self, request_id: str, model_name: str, predicted_state: Dict,
                         confidence: float = 1.0, processing_time: float = 0.0):
        """
        Submit model prediction to shared response file
        
        Args:
            request_id: Request identifier
            model_name: Name of the model
            predicted_state: Predicted state
            confidence: Prediction confidence
            processing_time: Time taken to process
        """
        with self.responses_lock:
            responses_data = self._read_json_file(self.responses_file)
            
            # Initialize request responses if not exists
            if request_id not in responses_data.get("responses", {}):
                responses_data.setdefault("responses", {})[request_id] = {}
            
            # Add model prediction
            responses_data["responses"][request_id][model_name] = {
                "predicted_state": predicted_state,
                "confidence": confidence,
                "processing_time": processing_time,
                "submitted_at": time.time()
            }
            
            # Update metadata
            responses_data.setdefault("metadata", {})["total_responses"] = (
                responses_data["metadata"].get("total_responses", 0) + 1
            )
            
            self._write_json_file(self.responses_file, responses_data)
        
        # Remove request from model's queue (it's been processed)
        self.queue_manager.remove_request_from_queue(model_name, request_id)
        
        print(f"✅ {model_name} submitted prediction for {request_id[:8]}")
    
    # ===== MODEL HEALTH MONITORING =====
    
    def register_model(self, model_name: str, status: str = "ready"):
        """
        Register/update model health status
        
        Args:
            model_name: Name of the model
            status: Model status (ready, active, stopped)
        """
        with self.coordination_lock:
            coord_data = self._read_json_file(self.coordination_file)
            
            coord_data.setdefault("model_health", {})[model_name] = {
                "status": status,
                "last_seen": time.time()
            }
            
            self._write_json_file(self.coordination_file, coord_data)
        
        if status == "ready":
            print(f"📋 Registered {model_name} as {status}")
    
    def get_active_models(self) -> List[str]:
        """
        Get list of currently active models
        
        Returns:
            List of active model names
        """
        with self.coordination_lock:
            coord_data = self._read_json_file(self.coordination_file)
            
            current_time = time.time()
            active_models = []
            
            for model_name, health_data in coord_data.get("model_health", {}).items():
                # Model active if seen within last 60 seconds
                if (current_time - health_data.get("last_seen", 0)) < 60:
                    active_models.append(model_name)
            
            return active_models
    
    # ===== CLEANUP AND MAINTENANCE =====
    
    def cleanup_expired_requests(self, max_age: float = 300.0):
        """
        Clean up expired requests from all queues and response files
        
        Args:
            max_age: Maximum age in seconds before cleanup
        """
        current_time = time.time()
        
        # Clean up individual queues
        queue_cleanup_results = self.queue_manager.cleanup_all_queues(max_age)
        
        # Clean up shared responses file
        with self.responses_lock:
            responses_data = self._read_json_file(self.responses_file)
            
            original_count = len(responses_data.get("responses", {}))
            
            # Remove old responses
            cutoff_time = current_time - max_age
            responses_data["responses"] = {
                req_id: resp_data for req_id, resp_data in responses_data.get("responses", {}).items()
                if any(
                    resp.get("submitted_at", 0) > cutoff_time 
                    for resp in resp_data.values()
                )
            }
            
            responses_cleaned = original_count - len(responses_data["responses"])
            
            if responses_cleaned > 0:
                self._write_json_file(self.responses_file, responses_data)
        
        # Clean up coordination file
        with self.coordination_lock:
            coord_data = self._read_json_file(self.coordination_file)
            
            # Clean old completed requests
            completed_requests = coord_data.get("completed_requests", [])
            original_completed = len(completed_requests)
            
            coord_data["completed_requests"] = [
                req for req in completed_requests
                if req.get("completed_at", 0) > cutoff_time
            ]
            
            completed_cleaned = original_completed - len(coord_data["completed_requests"])
            
            # Update cleanup timestamp
            coord_data.setdefault("system_metadata", {})["last_cleanup"] = current_time
            
            if completed_cleaned > 0:
                self._write_json_file(self.coordination_file, coord_data)
        
        # Report cleanup results
        total_queue_cleaned = sum(queue_cleanup_results.values())
        if total_queue_cleaned > 0 or responses_cleaned > 0 or completed_cleaned > 0:
            print(f"🧹 Cleanup results: {total_queue_cleaned} queue requests, "
                  f"{responses_cleaned} responses, {completed_cleaned} completed requests")
    
    # ===== SYSTEM STATUS AND MONITORING =====
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        # Queue status
        queue_status = self.queue_manager.get_queue_status()
        queue_depths = self.queue_manager.get_queue_depths()
        
        # Active models
        active_models = self.get_active_models()
        
        # Response file status
        with self.responses_lock:
            responses_data = self._read_json_file(self.responses_file)
            total_responses = len(responses_data.get("responses", {}))
        
        # Coordination status
        with self.coordination_lock:
            coord_data = self._read_json_file(self.coordination_file)
            active_requests = len(coord_data.get("active_requests", []))
            completed_requests = len(coord_data.get("completed_requests", []))
        
        return {
            "active_models": active_models,
            "queue_depths": queue_depths,
            "queue_status": queue_status,
            "active_requests": active_requests,
            "completed_requests": completed_requests,
            "total_responses": total_responses,
            "system_health": {
                "queues_operational": len(queue_depths) == len(self.model_names),
                "responses_file_exists": self.responses_file.exists(),
                "coordination_file_exists": self.coordination_file.exists()
            }
        }
    
    # ===== STATE CONVERSION UTILITIES (Preserve compatibility) =====
    
    def create_enhanced_state(self, holding_registers: Dict, coils: Dict, current_time: int = 0) -> Dict:
        """
        Create enhanced state format - preserved from original StateStore
        """
        enhanced_state = {
            "holding_registers": {},
            "coils": {},
            "metadata": {
                "timestamp": time.time(),
                "sequence_id": current_time
            }
        }
        
        # Enhanced holding registers with temporal info
        for i in range(39):
            reg_data = holding_registers.get(i, {"value": 0, "last_changed": 0, "change_count": 0})
            enhanced_state["holding_registers"][str(i)] = {
                "value": reg_data["value"],
                "last_changed_seconds": reg_data.get("last_changed", 0),
                "total_changes": reg_data.get("change_count", 0)
            }
        
        # Enhanced coils with temporal info  
        for i in range(19):
            coil_data = coils.get(i, {"value": 0, "last_changed": 0, "change_count": 0})
            enhanced_state["coils"][str(i)] = {
                "value": coil_data["value"], 
                "last_changed_seconds": coil_data.get("last_changed", 0),
                "total_changes": coil_data.get("change_count", 0)
            }
            
        return enhanced_state


# ===== UTILITY FUNCTIONS (Preserve from original state_store.py) =====

def enhanced_to_display_state(enhanced_state: Dict) -> Dict:
    """Convert enhanced state to display format (no temporal info) - PRESERVED"""
    display_state = {
        "holding_registers": {},
        "coils": {}
    }
    
    # Use string keys to match how enhanced_state is created
    for i in range(39):
        key = str(i)  # Convert to string to match enhanced state keys
        if key in enhanced_state["holding_registers"]:
            display_state["holding_registers"][i] = enhanced_state["holding_registers"][key]["value"]
        else:
            display_state["holding_registers"][i] = 0  # Default value
    
    for i in range(19):
        key = str(i)  # Convert to string to match enhanced state keys
        if key in enhanced_state["coils"]:
            display_state["coils"][i] = enhanced_state["coils"][key]["value"]
        else:
            display_state["coils"][i] = 0  # Default value
    
    return display_state

def generate_state_output_summary(holding_registers: Dict, coils: Dict) -> str:
    """Generate display summary - PRESERVED"""
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


# Legacy alias for backward compatibility
StateStore = DistributedStateStore