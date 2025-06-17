# shared/temporal_state.py
"""
Real-time temporal state management for ensemble system
"""

import time
from typing import Dict, Any

class TemporalStateManager:
    """Manages state with proper real-time temporal tracking"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_prediction_time = self.start_time
    
    def create_initial_state(self, holding_register_values: Dict[int, int], 
                           coil_values: Dict[int, int]) -> Dict:
        """Create initial state with last_changed=0 for all"""
        current_time = time.time()
        self.start_time = current_time
        self.last_prediction_time = current_time
        
        state = {
            "holding_registers": {},
            "coils": {},
            "metadata": {
                "ensemble_start_time": self.start_time,
                "last_prediction_time": current_time
            }
        }
        
        # Initialize holding registers
        for i in range(39):
            state["holding_registers"][i] = {
                "value": holding_register_values.get(i, 0),
                "last_changed": 0,  # Seconds since last change
                "total_changes": 0
            }
        
        # Initialize coils
        for i in range(19):
            state["coils"][i] = {
                "value": coil_values.get(i, 0),
                "last_changed": 0,  # Seconds since last change
                "total_changes": 0
            }
        
        return state
    
    def advance_time_for_prediction(self, current_state: Dict) -> Dict:
        """Advance time for all registers/coils before making prediction"""
        current_time = time.time()
        elapsed_since_last_prediction = int(current_time - self.last_prediction_time)
        
        new_state = {
            "holding_registers": {},
            "coils": {},
            "metadata": current_state.get("metadata", {})
        }
        
        # Update metadata
        new_state["metadata"]["last_prediction_time"] = current_time
        
        # Advance time for holding registers
        for i in range(39):
            reg = current_state["holding_registers"][i]
            new_state["holding_registers"][i] = {
                "value": reg["value"],
                "last_changed": reg["last_changed"] + elapsed_since_last_prediction,
                "total_changes": reg["total_changes"]
            }
        
        # Advance time for coils
        for i in range(19):
            coil = current_state["coils"][i]
            new_state["coils"][i] = {
                "value": coil["value"],
                "last_changed": coil["last_changed"] + elapsed_since_last_prediction,
                "total_changes": coil["total_changes"]
            }
        
        self.last_prediction_time = current_time
        return new_state
    
    def apply_consensus_changes(self, current_state: Dict, 
                              consensus_prediction: Dict) -> Dict:
        """Apply consensus prediction and update temporal metadata"""
        new_state = {
            "holding_registers": {},
            "coils": {},
            "metadata": current_state["metadata"]
        }
        
        # Process holding registers
        for i in range(39):
            current_reg = current_state["holding_registers"][i]
            predicted_value = consensus_prediction["holding_registers"].get(i, current_reg["value"])
            
            if predicted_value != current_reg["value"]:
                # Value changed - reset last_changed, increment total_changes
                new_state["holding_registers"][i] = {
                    "value": predicted_value,
                    "last_changed": 0,  # Reset to 0 seconds
                    "total_changes": current_reg["total_changes"] + 1
                }
            else:
                # Value unchanged - keep current temporal state
                new_state["holding_registers"][i] = {
                    "value": predicted_value,
                    "last_changed": current_reg["last_changed"],
                    "total_changes": current_reg["total_changes"]
                }
        
        # Process coils
        for i in range(19):
            current_coil = current_state["coils"][i]
            predicted_value = consensus_prediction["coils"].get(i, current_coil["value"])
            
            if predicted_value != current_coil["value"]:
                # Value changed - reset last_changed, increment total_changes
                new_state["coils"][i] = {
                    "value": predicted_value,
                    "last_changed": 0,  # Reset to 0 seconds
                    "total_changes": current_coil["total_changes"] + 1
                }
            else:
                # Value unchanged - keep current temporal state
                new_state["coils"][i] = {
                    "value": predicted_value,
                    "last_changed": current_coil["last_changed"],
                    "total_changes": current_coil["total_changes"]
                }
        
        return new_state
    
    def convert_to_enhanced_format(self, state):
        """Convert temporal state to enhanced format for models"""
        enhanced = {"holding_registers": {}, "coils": {}, "metadata": {}}
        
        for i in range(39):
            reg = state["holding_registers"][i]
            enhanced["holding_registers"][str(i)] = {
                "value": reg["value"],
                "last_changed_seconds": reg["last_changed"],
                "total_changes": reg["total_changes"]
            }
        
        for i in range(19):
            coil = state["coils"][i]
            enhanced["coils"][str(i)] = {
                "value": coil["value"],
                "last_changed_seconds": coil["last_changed"],
                "total_changes": coil["total_changes"]
            }
        
        enhanced["metadata"] = {
            "timestamp": time.time(),
            "sequence_id": int(time.time() - state["metadata"]["ensemble_start_time"])
        }
        
        return enhanced