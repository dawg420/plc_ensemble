# models/llm_predictor.py
"""
LLM Model Predictor Notebook/Script
Monitors shared state store and provides LLM predictions using Unsloth
"""

import time
import sys
import re
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.state_store import StateStore, enhanced_to_display_state, generate_state_output_summary
from shared.utils import ModelTimer, safe_model_prediction, log_ensemble_event, parse_state_summary

class LLMPredictorService:
    """LLM Model Prediction Service using Unsloth"""
    
    def __init__(self, model_path="models/saved_models/state_model", 
                 store_path="shared_state.json"):
        self.model_name = "llm"
        self.model_path = model_path
        self.store = StateStore(store_path)
        self.model = None
        self.tokenizer = None
        self.is_running = False
        self.prediction_count = 0
        
        # Model parameters (matching your setup)
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        
        # Load model
        self._load_model()
        
        # Register with state store
        self.store.register_model(self.model_name, "ready")
        
    def _load_model(self):
        """Load the trained LLM model using Unsloth"""
        try:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
            
            log_ensemble_event("MODEL_LOADING", f"Loading LLM from {self.model_path}")
            
            # Load model and tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            
            # Set up chat template
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="llama-3.2",
            )
            
            log_ensemble_event("MODEL_LOADED", "LLM model loaded successfully")
            return True
            
        except Exception as e:
            log_ensemble_event("ERROR", f"Failed to load LLM model: {e}")
            return False
    
    def _create_state_summary_prompt(self, enhanced_state):
        """Create state summary prompt for LLM input"""
        # Convert enhanced state to display format for prompt
        display_state = enhanced_to_display_state(enhanced_state)
        
        # Create the full state summary including temporal information
        summary_lines = ["STATE SUMMARY"]
        
        # Holding Registers with temporal info
        summary_lines.append("Holding Registers:")
        summary_lines.append("Reg | Value | LastChange | Changes")
        
        for i in range(39):
            reg_data = enhanced_state["holding_registers"].get(str(i), {
                "value": 0, "last_changed_seconds": 0, "total_changes": 0
            })
            summary_lines.append(
                f"HR{i:02d} | {reg_data['value']:4d} | {reg_data['last_changed_seconds']:5d} | {reg_data['total_changes']}"
            )
        
        # Coils with temporal info
        summary_lines.append("Coils:")
        summary_lines.append("Coil | Value | LastChange | Changes")
        
        for i in range(19):
            coil_data = enhanced_state["coils"].get(str(i), {
                "value": 0, "last_changed_seconds": 0, "total_changes": 0
            })
            summary_lines.append(
                f"C{i:02d}  | {coil_data['value']:4d} | {coil_data['last_changed_seconds']:5d} | {coil_data['total_changes']}"
            )
        
        return "\n".join(summary_lines)
    
    def _extract_assistant_response(self, decoded_text):
        """Extract the assistant's response from decoded LLM output"""
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)"
        matches = re.findall(pattern, decoded_text, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    def _calculate_prediction_confidence(self, predicted_response):
        """Calculate confidence score based on response quality"""
        if not predicted_response:
            return 0.1
        
        # Check if response has proper format
        has_state_summary = "STATE SUMMARY" in predicted_response
        has_registers = "HR" in predicted_response
        has_coils = "C" in predicted_response
        
        # Base confidence on format completeness
        format_score = 0.0
        if has_state_summary:
            format_score += 0.4
        if has_registers:
            format_score += 0.3
        if has_coils:
            format_score += 0.3
        
        # Bonus for reasonable response length
        length_score = min(0.2, len(predicted_response) / 2000.0)
        
        total_confidence = min(0.95, format_score + length_score)
        return max(0.1, total_confidence)
    
    def process_prediction_request(self, request_data):
        """Process a single prediction request"""
        request_id = request_data["request_id"]
        current_state = request_data["current_state"]
        remaining_time = request_data["remaining_time"]
        
        if remaining_time <= 0:
            log_ensemble_event("TIMEOUT", f"Request expired before processing", request_id)
            return
        
        log_ensemble_event("PROCESSING", f"LLM processing request", request_id)
        
        with ModelTimer(self.model_name) as timer:
            try:
                # Create the input prompt
                state_prompt = self._create_state_summary_prompt(current_state)
                
                # Format messages for chat template
                messages = [
                    {"role": "user", "content": state_prompt}
                ]
                
                # Apply chat template and tokenize
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                
                # Generate response
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=2048,
                    use_cache=True,
                    temperature=0.01,
                    min_p=0.1,
                )
                
                # Decode the response
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                predicted_response = self._extract_assistant_response(decoded[0])
                
                if not predicted_response:
                    log_ensemble_event("ERROR", "LLM generated empty response", request_id)
                    return
                
                # Parse the predicted state
                predicted_state = parse_state_summary(predicted_response)
                
                if not predicted_state or not predicted_state.get("holding_registers"):
                    log_ensemble_event("ERROR", "LLM generated invalid state format", request_id)
                    return
                
                # Calculate confidence
                confidence = self._calculate_prediction_confidence(predicted_response)
                
                # Submit prediction to state store
                processing_time = time.time() - timer.start_time
                self.store.submit_prediction(
                    request_id=request_id,
                    model_name=self.model_name,
                    predicted_state=predicted_state,
                    confidence=confidence,
                    processing_time=processing_time
                )
                
                self.prediction_count += 1
                log_ensemble_event("SUCCESS", f"LLM prediction submitted (confidence: {confidence:.3f})", request_id)
                
            except Exception as e:
                log_ensemble_event("ERROR", f"LLM prediction error: {e}", request_id)
    
    def run_prediction_loop(self, poll_interval=1.0):
        """Main prediction loop"""
        log_ensemble_event("STARTUP", f"LLM prediction service starting (poll every {poll_interval}s)")
        self.is_running = True
        
        while self.is_running:
            try:
                # Get pending requests
                pending_requests = self.store.get_pending_requests(self.model_name)
                
                if pending_requests:
                    log_ensemble_event("POLLING", f"Found {len(pending_requests)} pending requests")
                    
                    # Process each request (LLM processes one at a time due to memory)
                    for request_data in pending_requests:
                        if not self.is_running:
                            break
                        self.process_prediction_request(request_data)
                        
                        # Clear GPU cache after each prediction
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Update model status
                self.store.register_model(self.model_name, "active")
                
                # Sleep before next poll
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                log_ensemble_event("SHUTDOWN", "LLM service interrupted by user")
                break
            except Exception as e:
                log_ensemble_event("ERROR", f"LLM service error: {e}")
                time.sleep(poll_interval * 2)  # Longer sleep on error
        
        self.is_running = False
        self.store.register_model(self.model_name, "stopped")
        log_ensemble_event("SHUTDOWN", f"LLM service stopped. Total predictions: {self.prediction_count}")
    
    def stop(self):
        """Stop the prediction service"""
        self.is_running = False

def run_llm_service_thread(model_path="saved_models/state_model/", 
                          store_path="shared_state.json"):
    """Run LLM service in a separate thread"""
    service = LLMPredictorService(model_path, store_path)
    service.run_prediction_loop()
    return service

if __name__ == "__main__":
    # Run as standalone service
    print("🤖 Starting LLM Prediction Service...")
    print("Press Ctrl+C to stop")
    
    try:
        service = LLMPredictorService()
        service.run_prediction_loop()
    except KeyboardInterrupt:
        print("\n👋 LLM service stopped by user")
    except Exception as e:
        print(f"❌ LLM service error: {e}")