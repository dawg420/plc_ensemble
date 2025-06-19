# orchestrator.py
"""
Updated Ensemble Orchestrator with real-time temporal tracking
"""

import time
import threading
import signal
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import json

# Local imports
from shared.state_store import StateStore, generate_state_output_summary, enhanced_to_display_state
from shared.consensus import AdaptiveConsensus, select_consensus_method
from shared.temporal_state import TemporalStateManager
from shared.utils import (
    format_state_for_display, 
    log_ensemble_event, 
    create_test_state,
    calculate_state_accuracy,
    parse_state_summary,
    update_state_with_transaction,
    weighted_register_similarity,
    register_similarity,
    ENSEMBLE_CONFIG
)

class EnsembleOrchestrator:
    """Main orchestrator for the ensemble prediction system with temporal tracking"""
    
    def __init__(self, store_path="shared_state.json"):
        self.store = StateStore(store_path)
        self.consensus_engine = AdaptiveConsensus()
        self.temporal_manager = TemporalStateManager()
        self.is_running = False
        self.prediction_count = 0
        self.model_processes = {}
        self.cleanup_thread = None
        self.current_state = None
        
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
        
        log_ensemble_event("STARTUP", "Ensemble Orchestrator initialized with temporal tracking")
    
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
        cmd = [sys.executable, "models/llm_predictor.py"]
        process = subprocess.Popen(cmd, cwd=Path.cwd())
        self.model_processes["llm"] = process
    
    def _start_hmm_service(self, config):
        """Start HMM service in separate process"""
        cmd = [sys.executable, "models/hmm_predictor.py"]
        process = subprocess.Popen(cmd, cwd=Path.cwd())
        self.model_processes["hmm"] = process
    
    def _start_ml_service(self, model_name, config):
        """Start ML service in separate process"""
        cmd = [sys.executable, "models/ml_predictors.py", model_name]
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
        """Main ensemble prediction method with fixed polling logic"""
        if timeout is None:
            timeout = ENSEMBLE_CONFIG["max_prediction_timeout"]
        
        # Publish prediction request
        request_id = self.store.publish_prediction_request(current_state, timeout)
        start_time = time.time()
        
        # FIXED: Capture active models at start and don't change during polling
        initial_active_models = self.store.get_active_models()
        expected_count = len(initial_active_models)
        
        log_ensemble_event("PREDICTION", 
            f"Ensemble prediction started (timeout: {timeout}s, expecting {expected_count} models: {initial_active_models})", 
            request_id)
        
        # Wait for predictions with periodic polling
        predictions = {}
        poll_interval = 0.2
        max_polls = int(timeout / poll_interval)
        
        for poll_count in range(max_polls):
            predictions = self.store.get_predictions(request_id)
            received_count = len(predictions) if predictions else 0
            elapsed = time.time() - start_time
            
            # FIXED: Use initial_active_models, not fresh fetch
            log_ensemble_event("POLLING_STATUS", 
                f"Poll {poll_count}: {received_count}/{expected_count} models responded, {elapsed:.1f}s elapsed", 
                request_id)
            
            # Debug: Show which models have responded
            if predictions:
                responding_models = list(predictions.keys())
                missing_models = [m for m in initial_active_models if m not in responding_models]
                log_ensemble_event("POLLING_DETAIL", 
                    f"Responding: {responding_models}, Missing: {missing_models}", 
                    request_id)
            
            # Check if we have enough predictions
            min_models = max(1, ENSEMBLE_CONFIG["min_models_for_consensus"])
            
            # OPTION 1: FORCE WAIT FOR ALL MODELS (recommended for LLM)
            if ENSEMBLE_CONFIG.get("wait_for_all_models", True):
                # Only proceed when ALL expected models have responded
                if received_count >= expected_count and expected_count > 0:
                    log_ensemble_event("CONSENSUS_TRIGGER", 
                        f"All {expected_count} models responded - proceeding with consensus", 
                        request_id)
                    break
                elif elapsed >= timeout * 0.95:  # Only exit at 95% of timeout
                    log_ensemble_event("CONSENSUS_TRIGGER", 
                        f"Timeout approaching ({elapsed:.1f}s) - proceeding with {received_count}/{expected_count} models", 
                        request_id)
                    break
            
            # OPTION 2: ORIGINAL LOGIC (faster but may exclude slow models)
            else:
                if received_count >= min_models:
                    # Proceed if we have most models or significant time has passed
                    if (received_count >= expected_count * 0.8 or 
                        elapsed >= timeout * 0.7):
                        log_ensemble_event("CONSENSUS_TRIGGER", 
                            f"Proceeding with {received_count}/{expected_count} models (80% threshold or 70% timeout)", 
                            request_id)
                        break
            
            time.sleep(poll_interval)
        
        # FIXED: Get final predictions with better error handling
        final_predictions = self.store.get_predictions(request_id)
        processing_time = time.time() - start_time
        
        # Debug the final state
        if final_predictions:
            log_ensemble_event("FINAL_PREDICTIONS", 
                f"Collected {len(final_predictions)} predictions: {list(final_predictions.keys())}", 
                request_id)
        else:
            log_ensemble_event("ERROR_DEBUG", 
                f"No final predictions found. Expected: {initial_active_models}, Polling found: {received_count}", 
                request_id)
        
        if not final_predictions:
            log_ensemble_event("ERROR", "No predictions received within timeout", request_id)
            return self._create_fallback_response(current_state, request_id, processing_time)
        
        log_ensemble_event("CONSENSUS_START", 
            f"Starting consensus with {len(final_predictions)} predictions after {processing_time:.2f}s", 
            request_id)
        
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
    
    def initialize_demo_state(self, scenario="normal"):
        """Initialize demo with proper temporal tracking"""
        scenarios = {
            "normal": {
                "holding_registers": {0: 1, 1: 1, 5: 1},
                "coils": {0: 1, 2: 1}
            },
            "complex": {
                "holding_registers": {i: i % 2 for i in range(10)},
                "coils": {i: i % 2 for i in range(8)}
            },
            "sparse": {
                "holding_registers": {15: 1, 30: 1},
                "coils": {10: 1, 18: 1}
            }
        }
        
        scenario_data = scenarios.get(scenario, scenarios["normal"])
        self.current_state = self.temporal_manager.create_initial_state(
            scenario_data["holding_registers"],
            scenario_data["coils"]
        )
        
        print(f"🚀 Initialized '{scenario}' scenario with proper temporal tracking")
        self._print_state_summary()
        return self.current_state
    
    def predict_next_state_with_temporal_update(self, delay_seconds=None):
        """Make prediction with proper time advancement"""
        if self.current_state is None:
            raise ValueError("No current state. Call initialize_demo_state() first.")
        
        # Optional: simulate time delay for testing
        if delay_seconds:
            time.sleep(delay_seconds)
            print(f"⏰ Simulated {delay_seconds} second delay")
        
        # Step 1: Advance time for all registers/coils
        print("⏳ Advancing time for all registers/coils...")
        time_advanced_state = self.temporal_manager.advance_time_for_prediction(self.current_state)
        
        # Step 2: Get ensemble prediction (existing logic)
        print("🤖 Getting ensemble predictions...")
        enhanced_state = self.temporal_manager.convert_to_enhanced_format(time_advanced_state)
        prediction_result = self.predict_next_state(enhanced_state)
        consensus_prediction = prediction_result["consensus_state"]
        
        # Step 3: Apply consensus changes with proper temporal updates
        print("📊 Applying consensus changes...")
        self.current_state = self.temporal_manager.apply_consensus_changes(
            time_advanced_state, consensus_prediction
        )
        
        # Step 4: Show results
        self._print_state_summary()
        self._print_prediction_metadata(prediction_result["metadata"])
        
        return self.current_state
    
    def evaluate_ensemble_on_test_data(self, dataset_path="modbus_output_with_time.csv", 
                                     test_rows=1000):
        """Evaluate ensemble accuracy on last N rows of dataset"""
        print(f"\n🔍 ENSEMBLE EVALUATION ON TEST DATA")
        print("=" * 60)
        print(f"Dataset: {dataset_path}")
        print(f"Test rows: {test_rows}")
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"❌ Dataset not found: {dataset_path}")
            return None
        
        # Use last N rows as test set
        test_df = df.tail(test_rows).reset_index(drop=True)
        train_df = df.iloc[:-test_rows]
        
        print(f"📊 Total rows: {len(df)}")
        print(f"Training rows: {len(train_df)}")
        print(f"Test rows: {len(test_df)}")
        
        # Initialize state from training data
        holding_registers = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)}
        coils = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
        
        print("Processing training data to get correct starting state...")
        for idx, row in train_df.iterrows():
            update_state_with_transaction(row, idx, holding_registers, coils)
        
        # Evaluation on test set
        print("Starting ensemble evaluation on test set...")
        results = []
        total_register_similarity = 0.0
        total_weighted_similarity = 0.0
        exact_match_count = 0
        
        for idx, row in test_df.iterrows():
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(test_df)} test samples...")
            
            # Current state
            current_state = {
                "holding_registers": copy.deepcopy(holding_registers),
                "coils": copy.deepcopy(coils)
            }
            
            # Update to get expected next state
            update_state_with_transaction(row, len(train_df) + idx, holding_registers, coils)
            expected_next_state = {
                "holding_registers": copy.deepcopy(holding_registers),
                "coils": copy.deepcopy(coils)
            }
            
            # Convert to enhanced format for ensemble
            enhanced_current_state = self.store.create_enhanced_state(
                current_state["holding_registers"],
                current_state["coils"],
                current_time=len(train_df) + idx
            )
            
            # Get ensemble prediction
            try:
                prediction_result = self.predict_next_state(enhanced_current_state, timeout=5.0)
                predicted_state = prediction_result["consensus_state"]
                agreement_score = prediction_result["metadata"]["agreement_score"]
                participating_models = prediction_result["metadata"]["participating_models"]
            except Exception as e:
                print(f"Prediction error at test sample {idx}: {e}")
                predicted_state = enhanced_to_display_state(enhanced_current_state)
                agreement_score = 0.0
                participating_models = []
            
            # Calculate metrics
            reg_sim = register_similarity(expected_next_state, predicted_state)
            weighted_sim = weighted_register_similarity(expected_next_state, predicted_state, 
                                                      zero_weight=0.1, nonzero_weight=1.0)
            exact_match = weighted_sim == 1.0
            
            if exact_match:
                exact_match_count += 1
            
            results.append({
                "index": idx,
                "register_similarity": reg_sim,
                "weighted_similarity": weighted_sim,
                "exact_match": exact_match,
                "agreement_score": agreement_score,
                "participating_models": participating_models
            })
            
            total_register_similarity += reg_sim
            total_weighted_similarity += weighted_sim
        
        # Calculate final metrics
        avg_register_similarity = total_register_similarity / len(test_df)
        avg_weighted_similarity = total_weighted_similarity / len(test_df)
        exact_match_accuracy = (exact_match_count / len(test_df)) * 100
        
        # Print results
        print(f"\n✅ ENSEMBLE EVALUATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Test Results:")
        print(f"   Average Register Similarity: {avg_register_similarity:.4f}")
        print(f"   Average Weighted Similarity: {avg_weighted_similarity:.4f}")
        print(f"   Exact Match Accuracy: {exact_match_accuracy:.2f}%")
        print(f"   Exact Matches: {exact_match_count}/{len(test_df)}")
        
        metrics = {
            "avg_register_similarity": avg_register_similarity,
            "avg_weighted_similarity": avg_weighted_similarity,
            "exact_match_accuracy": exact_match_accuracy,
            "exact_match_count": exact_match_count,
            "total_samples": len(test_df)
        }
        
        return results, metrics
    
    def _print_state_summary(self):
        """Print current state with temporal information"""
        print("\n📊 CURRENT STATE (with temporal tracking)")
        print("=" * 60)
        
        # Show active registers
        active_regs = [(i, reg) for i, reg in self.current_state["holding_registers"].items() 
                      if reg["value"] != 0 or reg["total_changes"] > 0]
        
        if active_regs:
            print("📈 Holding Registers:")
            print("Reg | Value | Last Changed (s) | Total Changes")
            print("-" * 45)
            for i, reg in active_regs
                print(f"HR{i:02d} | {reg['value']:5d} | {reg['last_changed']:12d} | {reg['total_changes']:5d}")
            if len(active_regs) > 10:
                print(f"... and {len(active_regs) - 10} more active registers")
        else:
            print("📈 Holding Registers: All zero, no changes")
        
        # Show active coils
        active_coils = [(i, coil) for i, coil in self.current_state["coils"].items() 
                       if coil["value"] != 0 or coil["total_changes"] > 0]
        
        if active_coils:
            print("\n⚡ Coils:")
            print("Coil | Value | Last Changed (s) | Total Changes")
            print("-" * 45)
            for i, coil in active_coils:
                print(f"C{i:02d}  | {coil['value']:5d} | {coil['last_changed']:12d} | {coil['total_changes']:5d}")
        else:
            print("⚡ Coils: All zero, no changes")
    
    def _print_prediction_metadata(self, metadata):
        """Print prediction metadata"""
        print(f"\n🎯 Prediction Metadata:")
        print(f"   Models: {', '.join(metadata['participating_models'])}")
        print(f"   Agreement: {metadata['agreement_score']:.2f}")
        print(f"   Processing Time: {metadata['processing_time']:.3f}s")
        print(f"   Method: {metadata['consensus_method']}")
    
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
        print("\n🎭 ENSEMBLE PREDICTION SYSTEM")
        print("=" * 50)
        
        self.is_running = True
        self.start_cleanup_thread()
        
        try:
            while True:
                print("\nOptions:")
                print("1. Test prediction with sample state")
                print("2. Evaluate models with test data")
                print("3. Show ensemble status")
                print("4. Show model performance")
                print("5. Quit")
                
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == "1":
                    self._demo_interactive_prediction()
                elif choice == "2":
                    self._demo_test_data_evaluation()
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
    
    def _demo_interactive_prediction(self):
        """Interactive prediction loop with temporal advancement"""
        print("\n🎯 INTERACTIVE PREDICTION")
        print("=" * 40)
        
        # Choose scenario
        scenarios = ["normal", "complex", "sparse"]
        print("Available scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i}. {scenario.capitalize()}")
        
        try:
            choice = int(input("Choose scenario (1-3): ")) - 1
            if 0 <= choice < len(scenarios):
                scenario = scenarios[choice]
                self.initialize_demo_state(scenario)
                
                # Interactive prediction loop
                while True:
                    print("\n" + "-" * 50)
                    print("Options:")
                    print("1. Predict next state")
                    print("2. Predict with time delay")
                    print("3. Return to main menu")
                    
                    pred_choice = input("Enter choice (1-3): ").strip()
                    
                    if pred_choice == "1":
                        print("\n🔄 Predicting next state...")
                        self.predict_next_state_with_temporal_update()
                    elif pred_choice == "2":
                        try:
                            delay = int(input("Enter delay in seconds (1-10): "))
                            if 1 <= delay <= 10:
                                print(f"\n🔄 Predicting next state with {delay}s delay...")
                                self.predict_next_state_with_temporal_update(delay_seconds=delay)
                            else:
                                print("Invalid delay. Please enter 1-10 seconds.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    elif pred_choice == "3":
                        break
                    else:
                        print("Invalid choice.")
                        
        except (ValueError, IndexError):
            print("Invalid scenario choice")
    
    def _demo_test_data_evaluation(self):
        """Evaluate ensemble on test data"""
        print("\n📊 TEST DATA EVALUATION")
        print("=" * 40)
        
        try:
            test_rows = int(input("Enter number of test rows (default 1000): ") or "1000")
            if test_rows < 100:
                test_rows = 100
                print(f"Minimum 100 rows required. Using {test_rows} rows.")
            elif test_rows > 5000:
                test_rows = 5000
                print(f"Maximum 5000 rows allowed. Using {test_rows} rows.")
            
            results, metrics = self.evaluate_ensemble_on_test_data(test_rows=test_rows)
            
            if results:
                print("\n📈 Additional Statistics:")
                avg_agreement = sum(r["agreement_score"] for r in results) / len(results)
                print(f"   Average Model Agreement: {avg_agreement:.3f}")
                
                # Model participation
                all_models = set()
                for r in results:
                    all_models.update(r["participating_models"])
                print(f"   Models Participated: {', '.join(sorted(all_models))}")
                
        except ValueError:
            print("Invalid input. Using default 1000 test rows.")
            self.evaluate_ensemble_on_test_data(test_rows=1000)
        except Exception as e:
            print(f"Evaluation failed: {e}")
    
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
        time.sleep(15)
        
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