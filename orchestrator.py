# orchestrator.py
"""
Main Ensemble Orchestrator
Coordinates all models and provides the final ensemble predictions
"""

import time
import threading
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json

# Local imports
from shared.state_store import StateStore, generate_state_output_summary, enhanced_to_display_state
from shared.consensus import AdaptiveConsensus, select_consensus_method
from shared.utils import (
    format_state_for_display, 
    log_ensemble_event, 
    create_test_state,
    calculate_state_accuracy,
    ENSEMBLE_CONFIG
)

class EnsembleOrchestrator:
    """Main orchestrator for the ensemble prediction system"""
    
    def __init__(self, store_path="shared_state.json"):
        self.store = StateStore(store_path)
        self.consensus_engine = AdaptiveConsensus()
        self.is_running = False
        self.prediction_count = 0
        self.model_processes = {}
        self.cleanup_thread = None
        
        # Performance tracking
        self.ensemble_history = []
        self.model_performance = {
            "llm": [],
            "hmm": [],
            "lstm": [],
            "xgboost": [],
            "random_forest": []
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        log_ensemble_event("STARTUP", "Ensemble Orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        log_ensemble_event("SHUTDOWN", f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start_model_services(self, model_configs: Dict):
        """Start individual model prediction services"""
        log_ensemble_event("STARTUP", "Starting model services...")
        
        for model_name, config in model_configs.items():
            if config.get("enabled", True):
                try:
                    if model_name == "llm":
                        self._start_llm_service(config)
                    elif model_name == "hmm":
                        self._start_hmm_service(config)
                    elif model_name in ["xgboost", "random_forest", "lstm"]:
                        self._start_ml_service(model_name, config)
                    
                    log_ensemble_event("STARTUP", f"Started {model_name} service")
                    time.sleep(1)  # Stagger startups
                    
                except Exception as e:
                    log_ensemble_event("ERROR", f"Failed to start {model_name}: {e}")
        
        # Wait for models to initialize
        time.sleep(3)
        active_models = self.store.get_active_models()
        log_ensemble_event("STARTUP", f"Active models: {active_models}")
    
    def _start_llm_service(self, config):
        """Start LLM service in separate process"""
        cmd = [
            sys.executable, "models/llm_predictor.py"
        ]
        process = subprocess.Popen(cmd, cwd=Path.cwd())
        self.model_processes["llm"] = process
    
    def _start_hmm_service(self, config):
        """Start HMM service in separate process"""
        cmd = [
            sys.executable, "models/hmm_predictor.py"
        ]
        process = subprocess.Popen(cmd, cwd=Path.cwd())
        self.model_processes["hmm"] = process
    
    def _start_ml_service(self, model_name, config):
        """Start ML service in separate process"""
        cmd = [
            sys.executable, "models/ml_predictors.py", model_name
        ]
        if "model_path" in config:
            cmd.extend(["--model_path", config["model_path"]])
        
        process = subprocess.Popen(cmd, cwd=Path.cwd())
        self.model_processes[model_name] = process
    
    def start_cleanup_thread(self):
        """Start cleanup thread for expired requests"""
        def cleanup_worker():
            while self.is_running:
                try:
                    self.store.cleanup_expired_requests(max_age=ENSEMBLE_CONFIG["cleanup_interval"])
                    time.sleep(60)  # Cleanup every minute
                except Exception as e:
                    log_ensemble_event("ERROR", f"Cleanup error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        log_ensemble_event("STARTUP", "Cleanup thread started")
    
    def predict_next_state(self, current_state: Dict, timeout: float = None) -> Dict:
        """
        Main ensemble prediction method
        
        Args:
            current_state: Enhanced state dictionary with temporal info
            timeout: Maximum time to wait for predictions (default: from config)
            
        Returns:
            Dictionary with consensus prediction and metadata
        """
        if timeout is None:
            timeout = ENSEMBLE_CONFIG["max_prediction_timeout"]
        
        # Publish prediction request
        request_id = self.store.publish_prediction_request(current_state, timeout)
        start_time = time.time()
        
        log_ensemble_event("PREDICTION", f"Ensemble prediction started (timeout: {timeout}s)", request_id)
        
        # Wait for predictions with periodic polling
        predictions = {}
        poll_interval = 0.2
        max_polls = int(timeout / poll_interval)
        
        for poll_count in range(max_polls):
            predictions = self.store.get_predictions(request_id)
            
            if predictions:
                active_models = self.store.get_active_models()
                received_count = len(predictions)
                expected_count = len(active_models)
                
                # Log progress
                if poll_count % 5 == 0:  # Every second
                    log_ensemble_event("POLLING", 
                        f"Received {received_count}/{expected_count} predictions", request_id)
                
                # Check if we have enough predictions
                min_models = max(1, ENSEMBLE_CONFIG["min_models_for_consensus"])
                if received_count >= min_models:
                    # Check if we should wait for more or proceed
                    elapsed = time.time() - start_time
                    
                    # Proceed if we have most models or significant time has passed
                    if (received_count >= expected_count * 0.8 or 
                        elapsed >= timeout * 0.7):
                        break
            
            time.sleep(poll_interval)
        
        # Final prediction collection
        final_predictions = self.store.get_predictions(request_id)
        processing_time = time.time() - start_time
        
        if not final_predictions:
            log_ensemble_event("ERROR", "No predictions received within timeout", request_id)
            return self._create_fallback_response(current_state, request_id, processing_time)
        
        # Generate consensus
        consensus_result = self._generate_consensus(final_predictions, current_state)
        
        # Mark request as complete
        self.store.mark_request_complete(request_id, consensus_result)
        
        # Track performance
        self._track_performance(final_predictions, consensus_result, processing_time)
        
        # Prepare final response
        response = {
            "consensus_state": consensus_result["consensus_state"],
            "metadata": {
                "request_id": request_id,
                "processing_time": processing_time,
                "participating_models": consensus_result["participating_models"],
                "agreement_score": consensus_result["agreement_score"],
                "consensus_method": consensus_result["method"],
                "model_weights": consensus_result.get("weights_used", {}),
                "prediction_count": self.prediction_count
            }
        }
        
        self.prediction_count += 1
        
        log_ensemble_event("SUCCESS", 
            f"Ensemble prediction complete ({len(final_predictions)} models, "
            f"agreement: {consensus_result['agreement_score']:.2f})", request_id)
        
        return response
    
    def _generate_consensus(self, predictions: Dict, current_state: Dict) -> Dict:
        """Generate consensus from model predictions"""
        
        # Select consensus method
        num_models = len(predictions)
        has_confidence = all("confidence" in pred for pred in predictions.values())
        
        consensus_method = select_consensus_method(
            agreement_threshold=ENSEMBLE_CONFIG["agreement_threshold"],
            num_models=num_models,
            has_confidence=has_confidence
        )
        
        log_ensemble_event("CONSENSUS", f"Using {consensus_method} consensus with {num_models} models")
        
        # Apply selected consensus method
        if consensus_method == "confidence_weighted":
            return self.consensus_engine.confidence_weighted_consensus(predictions)
        elif consensus_method == "hierarchical":
            return self.consensus_engine.hierarchical_consensus(predictions)
        else:
            return self.consensus_engine.weighted_majority_vote(predictions)
    
    def _create_fallback_response(self, current_state: Dict, request_id: str, processing_time: float) -> Dict:
        """Create fallback response when no predictions are available"""
        # Use current state as fallback (no change)
        fallback_state = enhanced_to_display_state(current_state)
        
        return {
            "consensus_state": fallback_state,
            "metadata": {
                "request_id": request_id,
                "processing_time": processing_time,
                "participating_models": [],
                "agreement_score": 0.0,
                "consensus_method": "fallback",
                "model_weights": {},
                "prediction_count": self.prediction_count,
                "is_fallback": True
            }
        }
    
    def _track_performance(self, predictions: Dict, consensus: Dict, processing_time: float):
        """Track model and ensemble performance"""
        
        # Track individual model performance against consensus
        consensus_state = consensus["consensus_state"]
        
        for model_name, pred_data in predictions.items():
            predicted_state = pred_data["predicted_state"]
            
            # Calculate accuracy against consensus
            accuracy = calculate_state_accuracy(consensus_state, predicted_state)
            
            # Update performance tracking
            if model_name.lower() in self.model_performance:
                self.model_performance[model_name.lower()].append({
                    "accuracy": accuracy,
                    "confidence": pred_data.get("confidence", 1.0),
                    "processing_time": pred_data.get("processing_time", 0.0),
                    "timestamp": time.time()
                })
                
                # Update consensus engine with performance
                self.consensus_engine.update_performance(model_name.lower(), accuracy)
        
        # Track ensemble performance
        self.ensemble_history.append({
            "processing_time": processing_time,
            "num_models": len(predictions),
            "agreement_score": consensus["agreement_score"],
            "consensus_method": consensus["method"],
            "timestamp": time.time()
        })
        
        # Keep only recent history
        max_history = 100
        for model_name in self.model_performance:
            if len(self.model_performance[model_name]) > max_history:
                self.model_performance[model_name] = self.model_performance[model_name][-max_history:]
        
        if len(self.ensemble_history) > max_history:
            self.ensemble_history = self.ensemble_history[-max_history:]
    
    def get_ensemble_status(self) -> Dict:
        """Get current ensemble system status"""
        active_models = self.store.get_active_models()
        current_weights = self.consensus_engine.get_current_weights()
        
        # Calculate recent performance metrics
        recent_performance = {}
        for model_name, history in self.model_performance.items():
            if history:
                recent_history = history[-20:]  # Last 20 predictions
                recent_performance[model_name] = {
                    "avg_accuracy": sum(h["accuracy"] for h in recent_history) / len(recent_history),
                    "avg_confidence": sum(h["confidence"] for h in recent_history) / len(recent_history),
                    "avg_processing_time": sum(h["processing_time"] for h in recent_history) / len(recent_history),
                    "prediction_count": len(history)
                }
        
        # Calculate ensemble metrics
        ensemble_metrics = {}
        if self.ensemble_history:
            recent_ensemble = self.ensemble_history[-20:]
            ensemble_metrics = {
                "avg_processing_time": sum(h["processing_time"] for h in recent_ensemble) / len(recent_ensemble),
                "avg_agreement": sum(h["agreement_score"] for h in recent_ensemble) / len(recent_ensemble),
                "avg_model_count": sum(h["num_models"] for h in recent_ensemble) / len(recent_ensemble),
                "total_predictions": len(self.ensemble_history)
            }
        
        return {
            "active_models": active_models,
            "model_weights": current_weights,
            "recent_performance": recent_performance,
            "ensemble_metrics": ensemble_metrics,
            "total_predictions": self.prediction_count,
            "is_running": self.is_running
        }
    
    def run_interactive_demo(self):
        """Run interactive demonstration of the ensemble system"""
        print("\n🎭 ENSEMBLE PREDICTION SYSTEM DEMO")
        print("=" * 50)
        
        self.is_running = True
        self.start_cleanup_thread()
        
        try:
            while True:
                print("\nOptions:")
                print("1. Test prediction with sample state")
                print("2. Test prediction with custom state")
                print("3. Show ensemble status")
                print("4. Show model performance")
                print("5. Quit")
                
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == "1":
                    self._demo_sample_prediction()
                elif choice == "2":
                    self._demo_custom_prediction()
                elif choice == "3":
                    self._demo_ensemble_status()
                elif choice == "4":
                    self._demo_model_performance()
                elif choice == "5":
                    break
                else:
                    print("Invalid choice, please try again.")
                    
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
        finally:
            self.stop()
    
    def _demo_sample_prediction(self):
        """Demo with predefined sample states"""
        scenarios = ["normal", "complex", "sparse"]
        
        print("\nSample scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i}. {scenario.capitalize()}")
        
        try:
            choice = int(input("Choose scenario (1-3): ")) - 1
            if 0 <= choice < len(scenarios):
                scenario = scenarios[choice]
                test_state = create_test_state(scenario)
                
                # Convert to enhanced state format
                enhanced_state = self.store.create_enhanced_state(
                    {i: {"value": v, "last_changed": 0, "change_count": 0} 
                     for i, v in test_state["holding_registers"].items()},
                    {i: {"value": v, "last_changed": 0, "change_count": 0} 
                     for i, v in test_state["coils"].items()},
                    current_time=1000
                )
                
                print(f"\n📊 Input State ({scenario}):")
                print(format_state_for_display(test_state))
                
                print("\n🔄 Running ensemble prediction...")
                result = self.predict_next_state(enhanced_state)
                
                print("\n📈 Ensemble Result:")
                print(format_state_for_display(result["consensus_state"]))
                
                metadata = result["metadata"]
                print(f"\n📋 Metadata:")
                print(f"   Models: {', '.join(metadata['participating_models'])}")
                print(f"   Agreement: {metadata['agreement_score']:.2f}")
                print(f"   Time: {metadata['processing_time']:.3f}s")
                print(f"   Method: {metadata['consensus_method']}")
                
        except (ValueError, IndexError):
            print("Invalid choice")
    
    def _demo_custom_prediction(self):
        """Demo with custom state input"""
        print("\nEnter custom state (simplified):")
        try:
            # Simple input for demo
            reg_input = input("Non-zero registers (format: reg1:val1,reg2:val2): ").strip()
            coil_input = input("Non-zero coils (format: coil1:val1,coil2:val2): ").strip()
            
            # Parse input
            test_state = {
                "holding_registers": {i: 0 for i in range(39)},
                "coils": {i: 0 for i in range(19)}
            }
            
            if reg_input:
                for pair in reg_input.split(","):
                    if ":" in pair:
                        reg, val = pair.split(":")
                        test_state["holding_registers"][int(reg)] = int(val)
            
            if coil_input:
                for pair in coil_input.split(","):
                    if ":" in pair:
                        coil, val = pair.split(":")
                        test_state["coils"][int(coil)] = int(val)
            
            # Convert to enhanced state
            enhanced_state = self.store.create_enhanced_state(
                {i: {"value": v, "last_changed": 0, "change_count": 0} 
                 for i, v in test_state["holding_registers"].items()},
                {i: {"value": v, "last_changed": 0, "change_count": 0} 
                 for i, v in test_state["coils"].items()},
                current_time=1000
            )
            
            print("\n📊 Input State:")
            print(format_state_for_display(test_state))
            
            print("\n🔄 Running ensemble prediction...")
            result = self.predict_next_state(enhanced_state)
            
            print("\n📈 Ensemble Result:")
            print(format_state_for_display(result["consensus_state"]))
            
        except Exception as e:
            print(f"Error processing input: {e}")
    
    def _demo_ensemble_status(self):
        """Show ensemble system status"""
        status = self.get_ensemble_status()
        
        print("\n🏥 ENSEMBLE SYSTEM STATUS")
        print("=" * 40)
        print(f"Active Models: {', '.join(status['active_models'])}")
        print(f"Total Predictions: {status['total_predictions']}")
        print(f"Running: {status['is_running']}")
        
        if status['ensemble_metrics']:
            metrics = status['ensemble_metrics']
            print(f"\n📊 Recent Performance:")
            print(f"   Avg Processing Time: {metrics['avg_processing_time']:.3f}s")
            print(f"   Avg Agreement: {metrics['avg_agreement']:.2f}")
            print(f"   Avg Models per Prediction: {metrics['avg_model_count']:.1f}")
        
        print(f"\n⚖️  Current Model Weights:")
        for model, weight in status['model_weights'].items():
            print(f"   {model}: {weight:.3f}")
    
    def _demo_model_performance(self):
        """Show individual model performance"""
        status = self.get_ensemble_status()
        
        print("\n📈 MODEL PERFORMANCE")
        print("=" * 40)
        
        for model_name, perf in status['recent_performance'].items():
            print(f"\n{model_name.upper()}:")
            print(f"   Accuracy: {perf['avg_accuracy']:.3f}")
            print(f"   Confidence: {perf['avg_confidence']:.3f}")
            print(f"   Speed: {perf['avg_processing_time']:.3f}s")
            print(f"   Predictions: {perf['prediction_count']}")
    
    def stop(self):
        """Stop the ensemble system"""
        self.is_running = False
        
        log_ensemble_event("SHUTDOWN", "Stopping ensemble orchestrator...")
        
        # Stop model processes
        for model_name, process in self.model_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                log_ensemble_event("SHUTDOWN", f"Stopped {model_name} service")
            except Exception as e:
                log_ensemble_event("ERROR", f"Error stopping {model_name}: {e}")
                try:
                    process.kill()
                except:
                    pass
        
        log_ensemble_event("SHUTDOWN", "Ensemble orchestrator stopped")

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "llm": {
        "enabled": True,
        "model_path": "models/saved_models/state_model"
    },
    "hmm": {
        "enabled": True,
        "model_path": "models/saved_models/hmm_model.pkl"
    },
    "xgboost": {
        "enabled": True,
        "model_path": "models/saved_models/xgboost_model.pkl"
    },
    "random_forest": {
        "enabled": True,
        "model_path": "models/saved_models/random_forest_model.pkl"
    },
    "lstm": {
        "enabled": True,
        "model_path": "models/saved_models/lstm_model.pkl"
    }
}

def main():
    """Main function to run the ensemble system"""
    print("🚀 Starting PLC Ensemble Prediction System")
    print("=" * 50)
    
    # Load configuration
    config_path = Path("ensemble_config.json")
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        print(f"📋 Loaded configuration from {config_path}")
    else:
        model_config = DEFAULT_MODEL_CONFIG
        print("📋 Using default configuration")
    
    # Create orchestrator
    orchestrator = EnsembleOrchestrator()
    
    try:
        # Start model services
        orchestrator.start_model_services(model_config)
        
        # Wait for models to be ready
        print("\n⏳ Waiting for models to initialize...")
        time.sleep(5)
        
        # Check which models are active
        active_models = orchestrator.store.get_active_models()
        print(f"✅ Active models: {active_models}")
        
        if len(active_models) == 0:
            print("❌ No models are active. Please check model services.")
            return
        
        # Run interactive demo
        orchestrator.run_interactive_demo()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        orchestrator.stop()

if __name__ == "__main__":
    main()