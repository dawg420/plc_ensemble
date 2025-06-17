# shared/consensus.py
import numpy as np
from collections import Counter
from typing import Dict, List, Any
import copy

class AdaptiveConsensus:
    """Ensemble consensus engine with adaptive weighting"""
    
    def __init__(self):
        # Weights based on your evaluation results
        # Calculated as: 0.7 * weighted_similarity + 0.3 * exact_match_rate
        self.model_weights = {
            "llm": 0.244,           # Best overall: 0.9898 weighted + 0.8950 exact
            "xgboost": 0.243,       # Close second: 0.9883 weighted + 0.8790 exact  
            "lstm": 0.220,          # Good: 0.9534 weighted + 0.6620 exact
            "random_forest": 0.155, # Moderate: 0.7897 weighted + 0.1920 exact
            "hmm": 0.137           # Baseline: 0.7663 weighted + 0.0130 exact
        }
        
        # Performance tracking for dynamic adjustment
        self.recent_performance = {model: [] for model in self.model_weights.keys()}
        self.performance_window = 50  # Track last 50 predictions
        
    def weighted_majority_vote(self, predictions: Dict, prediction_type: str = "state") -> Dict:
        """
        Perform weighted majority voting on predictions
        
        Args:
            predictions: Dict of {model_name: prediction_data}
            prediction_type: "state" for full state prediction, "register" for per-register voting
        """
        if not predictions:
            return self._create_empty_state()
        
        print(f"🗳️  Performing weighted consensus on {len(predictions)} predictions")
        
        # Extract states and weights
        model_states = {}
        active_weights = {}
        
        for model_name, pred_data in predictions.items():
            if model_name.lower() in self.model_weights:
                model_states[model_name] = pred_data["predicted_state"]
                active_weights[model_name] = self.model_weights[model_name.lower()]
                print(f"   {model_name}: weight={active_weights[model_name]:.3f}, confidence={pred_data.get('confidence', 1.0):.3f}")
        
        if not model_states:
            print("❌ No valid predictions found")
            return self._create_empty_state()
        
        # Normalize weights for active models
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v/total_weight for k, v in active_weights.items()}
        
        consensus_state = self._vote_on_registers_and_coils(model_states, normalized_weights)
        
        # Calculate agreement metrics
        agreement_score = self._calculate_agreement(model_states, consensus_state)
        
        result = {
            "consensus_state": consensus_state,
            "agreement_score": agreement_score,
            "participating_models": list(model_states.keys()),
            "weights_used": normalized_weights,
            "method": "weighted_majority"
        }
        
        print(f"✅ Consensus reached with {agreement_score:.2f} agreement")
        return result
    
    def _vote_on_registers_and_coils(self, model_states: Dict, weights: Dict) -> Dict:
        """Vote on each register and coil independently"""
        consensus = {"holding_registers": {}, "coils": {}}
        
        # Vote on holding registers (39 registers)
        for reg_idx in range(39):
            votes = []
            vote_weights = []
            
            for model_name, state in model_states.items():
                value = state.get("holding_registers", {}).get(reg_idx, 0)
                votes.append(value)
                vote_weights.append(weights[model_name])
            
            consensus["holding_registers"][reg_idx] = self._weighted_mode(votes, vote_weights)
        
        # Vote on coils (19 coils)  
        for coil_idx in range(19):
            votes = []
            vote_weights = []
            
            for model_name, state in model_states.items():
                value = state.get("coils", {}).get(coil_idx, 0)
                votes.append(value)
                vote_weights.append(weights[model_name])
            
            consensus["coils"][coil_idx] = self._weighted_mode(votes, vote_weights)
        
        return consensus
    
    def _weighted_mode(self, values: List, weights: List) -> int:
        """Calculate weighted mode of values"""
        if not values:
            return 0
        
        # Create weighted vote counts
        vote_counts = {}
        for value, weight in zip(values, weights):
            vote_counts[value] = vote_counts.get(value, 0) + weight
        
        # Return value with highest weighted count
        return max(vote_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_agreement(self, model_states: Dict, consensus_state: Dict) -> float:
        """Calculate how much models agree with consensus"""
        if not model_states:
            return 0.0
        
        total_positions = 39 + 19  # registers + coils
        agreements = []
        
        for model_name, state in model_states.items():
            agreement_count = 0
            
            # Check holding registers
            for reg_idx in range(39):
                model_val = state.get("holding_registers", {}).get(reg_idx, 0)
                consensus_val = consensus_state["holding_registers"][reg_idx]
                if model_val == consensus_val:
                    agreement_count += 1
            
            # Check coils
            for coil_idx in range(19):
                model_val = state.get("coils", {}).get(coil_idx, 0)
                consensus_val = consensus_state["coils"][coil_idx]
                if model_val == consensus_val:
                    agreement_count += 1
            
            agreement_ratio = agreement_count / total_positions
            agreements.append(agreement_ratio)
        
        return np.mean(agreements)
    
    def confidence_weighted_consensus(self, predictions: Dict) -> Dict:
        """Alternative consensus method using prediction confidence"""
        if not predictions:
            return self._create_empty_state()
        
        print(f"🎯 Performing confidence-weighted consensus")
        
        # Combine model weights with prediction confidence
        combined_weights = {}
        model_states = {}
        
        for model_name, pred_data in predictions.items():
            if model_name.lower() in self.model_weights:
                base_weight = self.model_weights[model_name.lower()]
                confidence = pred_data.get("confidence", 1.0)
                combined_weights[model_name] = base_weight * confidence
                model_states[model_name] = pred_data["predicted_state"]
                print(f"   {model_name}: base_weight={base_weight:.3f} × confidence={confidence:.3f} = {combined_weights[model_name]:.3f}")
        
        if not combined_weights:
            return self._create_empty_state()
        
        # Normalize combined weights
        total_weight = sum(combined_weights.values())
        normalized_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        consensus_state = self._vote_on_registers_and_coils(model_states, normalized_weights)
        agreement_score = self._calculate_agreement(model_states, consensus_state)
        
        return {
            "consensus_state": consensus_state,
            "agreement_score": agreement_score,
            "participating_models": list(model_states.keys()),
            "weights_used": normalized_weights,
            "method": "confidence_weighted"
        }
    
    def hierarchical_consensus(self, predictions: Dict, primary_models: List[str] = None) -> Dict:
        """Hierarchical consensus: primary models first, fallback to ensemble"""
        if primary_models is None:
            primary_models = ["llm", "xgboost"]  # Top performers
        
        primary_predictions = {}
        fallback_predictions = {}
        
        for model_name, pred_data in predictions.items():
            if model_name.lower() in [p.lower() for p in primary_models]:
                primary_predictions[model_name] = pred_data
            else:
                fallback_predictions[model_name] = pred_data
        
        print(f"🏛️  Hierarchical consensus: {len(primary_predictions)} primary, {len(fallback_predictions)} fallback")
        
        # Try primary models first
        if len(primary_predictions) >= 2:
            return self.weighted_majority_vote(primary_predictions)
        elif len(primary_predictions) == 1:
            # Single primary model - use its prediction with high confidence
            model_name, pred_data = list(primary_predictions.items())[0]
            return {
                "consensus_state": pred_data["predicted_state"],
                "agreement_score": 1.0,
                "participating_models": [model_name],
                "weights_used": {model_name: 1.0},
                "method": "single_primary"
            }
        else:
            # Fall back to ensemble
            return self.weighted_majority_vote(fallback_predictions)
    
    def update_performance(self, model_name: str, accuracy_score: float):
        """Update recent performance tracking for adaptive weighting"""
        if model_name.lower() in self.recent_performance:
            self.recent_performance[model_name.lower()].append(accuracy_score)
            
            # Keep only recent history
            if len(self.recent_performance[model_name.lower()]) > self.performance_window:
                self.recent_performance[model_name.lower()].pop(0)
            
            # Adapt weights based on recent performance
            self._adapt_weights()
    
    def _adapt_weights(self):
        """Dynamically adjust weights based on recent performance"""
        # Calculate recent average performance
        recent_averages = {}
        for model_name, scores in self.recent_performance.items():
            if len(scores) >= 10:  # Need at least 10 recent predictions
                recent_averages[model_name] = np.mean(scores[-20:])  # Last 20 predictions
        
        if len(recent_averages) >= 3:  # Need at least 3 models with history
            # Adjust weights slightly based on recent performance
            adjustment_factor = 0.1  # Conservative adjustment
            
            for model_name in recent_averages:
                recent_perf = recent_averages[model_name]
                base_weight = self.model_weights[model_name]
                
                # Adjust weight by ±10% based on recent performance vs expected
                expected_perf = base_weight  # Use base weight as expected performance proxy
                performance_ratio = recent_perf / expected_perf if expected_perf > 0 else 1.0
                
                adjustment = (performance_ratio - 1.0) * adjustment_factor
                new_weight = max(0.05, min(0.4, base_weight + adjustment))  # Bounded adjustment
                
                self.model_weights[model_name] = new_weight
            
            print(f"🔄 Adapted weights based on recent performance")
    
    def _create_empty_state(self) -> Dict:
        """Create empty state as fallback"""
        return {
            "consensus_state": {
                "holding_registers": {i: 0 for i in range(39)},
                "coils": {i: 0 for i in range(19)}
            },
            "agreement_score": 0.0,
            "participating_models": [],
            "weights_used": {},
            "method": "empty_fallback"
        }
    
    def get_current_weights(self) -> Dict:
        """Get current model weights"""
        return copy.deepcopy(self.model_weights)
    
    def set_weights(self, new_weights: Dict):
        """Manually set model weights"""
        for model_name, weight in new_weights.items():
            if model_name.lower() in self.model_weights:
                self.model_weights[model_name.lower()] = weight
        print(f"🎛️  Updated model weights: {self.model_weights}")

# Utility function for consensus selection
def select_consensus_method(agreement_threshold: float = 0.7, 
                          num_models: int = 5,
                          has_confidence: bool = True) -> str:
    """Select appropriate consensus method based on situation"""
    if num_models <= 2:
        return "hierarchical"
    elif has_confidence and agreement_threshold < 0.5:
        return "confidence_weighted"
    else:
        return "weighted_majority"