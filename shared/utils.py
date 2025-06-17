# shared/utils_fixed.py
import time
import re
import copy
from typing import Dict, Any

def update_state_with_transaction(row, idx, holding_registers, coils):
    """Update state tracking actual value changes (your existing function)"""
    func_code = row['func_code']
    request = row['request_payload']
    response = row['response_payload']

    try:
        req_bytes = bytes.fromhex(request.replace('\\x', ''))
        resp_bytes = bytes.fromhex(response.replace('\\x', ''))

        if func_code == 3:  # Read Holding Registers
            if len(req_bytes) >= 4:
                start_addr = int.from_bytes(req_bytes[0:2], byteorder='big')
                quantity = int.from_bytes(req_bytes[2:4], byteorder='big')

                if len(resp_bytes) > 0:
                    for i in range(quantity):
                        offset = 1 + i*2
                        if offset + 1 < len(resp_bytes) and start_addr + i < 39:
                            register_idx = start_addr + i
                            value = int.from_bytes(resp_bytes[offset:offset+2], byteorder='big')

                            if holding_registers[register_idx]["value"] != value:
                                holding_registers[register_idx]["change_count"] += 1
                                holding_registers[register_idx]["last_changed"] = idx
                                holding_registers[register_idx]["value"] = value

        elif func_code == 1:  # Read Coils
            if len(req_bytes) >= 4:
                start_addr = int.from_bytes(req_bytes[0:2], byteorder='big')
                quantity = int.from_bytes(req_bytes[2:4], byteorder='big')

                if len(resp_bytes) > 0:
                    for i in range(quantity):
                        byte_idx = 1 + (i // 8)
                        bit_idx = i % 8

                        if byte_idx < len(resp_bytes) and start_addr + i < 19:
                            coil_idx = start_addr + i
                            value = (resp_bytes[byte_idx] >> bit_idx) & 1

                            if coils[coil_idx]["value"] != value:
                                coils[coil_idx]["change_count"] += 1
                                coils[coil_idx]["last_changed"] = idx
                                coils[coil_idx]["value"] = value
    except Exception as e:
        print(f"Error updating state: {e}")
        pass

def state_to_features_for_training(holding_registers, coils, current_idx=0):
    """Convert state to feature vector for TRAINING (includes temporal info)"""
    features = []
    
    # Holding registers: value, time_since_change, change_count
    for i in range(39):
        reg = holding_registers[i]
        features.append(reg["value"])                           
        features.append(current_idx - reg["last_changed"])      
        features.append(reg["change_count"])                    
    
    # Coils: value, time_since_change, change_count  
    for i in range(19):
        coil = coils[i]
        features.append(coil["value"])                          
        features.append(current_idx - coil["last_changed"])     
        features.append(coil["change_count"])                   
        
    return features

def state_to_values_only(holding_registers, coils):
    """Convert state to values-only feature vector for PREDICTION"""
    values = []
    
    # Holding register values only
    for i in range(39):
        values.append(holding_registers[i]["value"])
    
    # Coil values only
    for i in range(19):
        values.append(coils[i]["value"])
        
    return values

def values_to_prediction_state(predicted_values, current_state, current_time):
    """
    FIXED: Convert predicted values to state with proper temporal metadata
    
    Args:
        predicted_values: List of 58 values (39 registers + 19 coils)
        current_state: Current state dict with temporal info
        current_time: Current time index (no longer used - real time handled elsewhere)
    
    Returns:
        State dict with values and UNCHANGED temporal metadata
    """
    new_state = {
        "holding_registers": {},
        "coils": {}
    }
    
    # Process holding registers (first 39 values)
    for i in range(39):
        predicted_value = max(0, int(round(predicted_values[i])))
        current_reg = current_state["holding_registers"][i]
        
        # FIXED: Don't update temporal metadata here - that's handled by TemporalStateManager
        new_state["holding_registers"][i] = {
            "value": predicted_value,
            "last_changed": current_reg["last_changed"],
            "change_count": current_reg["change_count"]
        }
    
    # Process coils (next 19 values)
    for i in range(19):
        predicted_value = max(0, min(1, int(round(predicted_values[39 + i]))))
        current_coil = current_state["coils"][i]
        
        # FIXED: Don't update temporal metadata here - that's handled by TemporalStateManager
        new_state["coils"][i] = {
            "value": predicted_value,
            "last_changed": current_coil["last_changed"],
            "change_count": current_coil["change_count"]
        }
    
    return new_state

def state_to_features(holding_registers, coils, current_idx=0):
    """Legacy function - redirects to training version for compatibility"""
    return state_to_features_for_training(holding_registers, coils, current_idx)

def features_to_state(features):
    """
    Legacy function for training compatibility
    Convert full feature vector back to state (used during training)
    """
    holding_registers = {}
    coils = {}
    
    # Holding registers: extract every 3rd element starting from each position
    for i in range(39):
        base_idx = i * 3
        holding_registers[i] = {
            "value": max(0, int(round(features[base_idx]))),      
            "last_changed": int(round(features[base_idx + 1])),   
            "change_count": max(0, int(round(features[base_idx + 2])))  
        }
    
    # Coils: start after holding registers (39*3 = 117)
    coils_start = 39 * 3
    for i in range(19):
        base_idx = coils_start + i * 3
        coils[i] = {
            "value": max(0, min(1, int(round(features[base_idx])))),     
            "last_changed": int(round(features[base_idx + 1])),
            "change_count": max(0, int(round(features[base_idx + 2])))
        }
        
    return holding_registers, coils

def parse_state_summary(state_text):
    """Parse a state summary text and extract register/coil values."""
    state = {"holding_registers": {}, "coils": {}}

    hr_pattern = r'HR(\d+)\s*\|\s*(\d+)'
    hr_matches = re.finditer(hr_pattern, state_text)
    for match in hr_matches:
        reg_idx = int(match.group(1))
        value = int(match.group(2))
        state["holding_registers"][reg_idx] = value

    coil_pattern = r'C(\d+)\s*\|\s*(\d+)'
    coil_matches = re.finditer(coil_pattern, state_text)
    for match in coil_matches:
        coil_idx = int(match.group(1))
        value = int(match.group(2))
        state["coils"][coil_idx] = value

    return state

def calculate_state_accuracy(expected_state, predicted_state, zero_weight=0.1, nonzero_weight=1.0):
    """Calculate weighted accuracy between two states"""
    total_weight = 0
    matched_weight = 0

    # Check holding registers
    for i in range(39):
        expected_val = expected_state.get("holding_registers", {}).get(i, 0)
        predicted_val = predicted_state.get("holding_registers", {}).get(i, 0)

        weight = zero_weight if expected_val == 0 else nonzero_weight
        total_weight += weight

        if expected_val == predicted_val:
            matched_weight += weight

    # Check coils
    for i in range(19):
        expected_val = expected_state.get("coils", {}).get(i, 0)
        predicted_val = predicted_state.get("coils", {}).get(i, 0)

        weight = zero_weight if expected_val == 0 else nonzero_weight
        total_weight += weight

        if expected_val == predicted_val:
            matched_weight += weight

    return matched_weight / total_weight if total_weight > 0 else 0.0

def weighted_register_similarity(expected_state, predicted_state, zero_weight=0.1, nonzero_weight=1.0):
    """Calculate weighted similarity where non-zero values have higher importance."""
    total_weight = 0
    matched_weight = 0

    # Check holding registers
    for i in range(39):
        expected_val = expected_state.get("holding_registers", {}).get(i, 0)
        predicted_val = predicted_state.get("holding_registers", {}).get(i, 0)

        weight = zero_weight if expected_val == 0 else nonzero_weight
        total_weight += weight

        if expected_val == predicted_val:
            matched_weight += weight

    # Check coils
    for i in range(19):
        expected_val = expected_state.get("coils", {}).get(i, 0)
        predicted_val = predicted_state.get("coils", {}).get(i, 0)

        weight = zero_weight if expected_val == 0 else nonzero_weight
        total_weight += weight

        if expected_val == predicted_val:
            matched_weight += weight

    return matched_weight / total_weight if total_weight > 0 else 0.0

def register_similarity(expected_state, predicted_state):
    """Calculate similarity based on register and coil values."""
    total_registers = 39 + 19
    correct_registers = 0

    for i in range(39):
        expected_val = expected_state.get("holding_registers", {}).get(i, 0)
        predicted_val = predicted_state.get("holding_registers", {}).get(i, 0)
        if expected_val == predicted_val:
            correct_registers += 1

    for i in range(19):
        expected_val = expected_state.get("coils", {}).get(i, 0)
        predicted_val = predicted_state.get("coils", {}).get(i, 0)
        if expected_val == predicted_val:
            correct_registers += 1

    return correct_registers / total_registers

def format_state_for_display(state_dict):
    """Format state for console display"""
    lines = []
    lines.append("📊 CURRENT STATE")
    lines.append("=" * 40)
    
    # Show only non-zero registers and coils for compact display
    non_zero_regs = [(i, val) for i, val in state_dict.get("holding_registers", {}).items() if val != 0]
    non_zero_coils = [(i, val) for i, val in state_dict.get("coils", {}).items() if val != 0]
    
    if non_zero_regs:
        lines.append(f"📈 Active Registers ({len(non_zero_regs)}/{39}):")
        for reg_idx, value in non_zero_regs[:10]:  # Show first 10
            lines.append(f"   HR{reg_idx:02d}: {value}")
        if len(non_zero_regs) > 10:
            lines.append(f"   ... and {len(non_zero_regs) - 10} more")
    else:
        lines.append("📈 Registers: All zero")
    
    if non_zero_coils:
        lines.append(f"⚡ Active Coils ({len(non_zero_coils)}/{19}):")
        for coil_idx, value in non_zero_coils:
            lines.append(f"   C{coil_idx:02d}: {value}")
    else:
        lines.append("⚡ Coils: All zero")
    
    return "\n".join(lines)

def model_health_check(model_name, last_response_time, max_response_time=5.0):
    """Check if a model is responding within acceptable time"""
    return last_response_time <= max_response_time

def create_test_state(scenario="normal"):
    """Create test states for development/debugging"""
    scenarios = {
        "normal": {
            "holding_registers": {0: 1, 1: 1, 5: 1},
            "coils": {0: 1, 2: 1}
        },
        "complex": {
            "holding_registers": {i: i % 2 for i in range(0, 20)},
            "coils": {i: i % 2 for i in range(0, 8)}
        },
        "sparse": {
            "holding_registers": {15: 1, 30: 1},
            "coils": {10: 1, 18: 1}
        }
    }
    
    if scenario not in scenarios:
        scenario = "normal"
    
    # Fill in missing registers/coils with zeros
    full_state = {
        "holding_registers": {i: 0 for i in range(39)},
        "coils": {i: 0 for i in range(19)}
    }
    
    # Update with scenario values
    full_state["holding_registers"].update(scenarios[scenario]["holding_registers"])
    full_state["coils"].update(scenarios[scenario]["coils"])
    
    return full_state

class ModelTimer:
    """Simple timer for model performance tracking"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"⏱️  {self.model_name} took {elapsed:.3f}s")
        return elapsed

def safe_model_prediction(predict_func, *args, **kwargs):
    """Safe wrapper for model predictions with error handling"""
    try:
        return predict_func(*args, **kwargs), None
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return None, str(e)

def log_ensemble_event(event_type, message, request_id=None):
    """Simple logging for ensemble events"""
    timestamp = time.strftime("%H:%M:%S")
    req_info = f"[{request_id[:8]}]" if request_id else ""
    print(f"[{timestamp}] {event_type} {req_info}: {message}")

# Constants for the ensemble system
ENSEMBLE_CONFIG = {
    "max_prediction_timeout": 3.0,
    "model_health_timeout": 5.0,
    "cleanup_interval": 300.0,  # 5 minutes
    "performance_window": 50,
    "min_models_for_consensus": 2,
    "agreement_threshold": 0.7
}