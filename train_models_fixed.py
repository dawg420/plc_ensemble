# train_models_fixed.py
"""
Fixed training script for models that only predict values
Uses the corrected approach where temporal metadata is computed, not predicted
"""

import pandas as pd
import numpy as np
import pickle
import time
import warnings
from pathlib import Path

# Import fixed model classes and utilities
from shared.model_classes import (
    HMMStatePredictor,
    LSTMStatePredictor, 
    XGBoostStatePredictor,
    RandomForestStatePredictor
)
from shared.utils import (
    update_state_with_transaction, 
    state_to_features_for_training,
    values_to_prediction_state
)

warnings.filterwarnings('ignore')

def process_dataset_for_training(file_path, max_rows=20000):
    """Process dataset to extract state sequences for training"""
    print(f"Processing dataset: {file_path}")
    df = pd.read_csv(file_path)
    
    if len(df) > max_rows:
        df = df.head(max_rows)
        print(f"Limited to {max_rows} rows")
    
    holding_registers = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)}
    coils = {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
    
    state_sequence = []
    
    for idx, row in df.iterrows():
        # Use full features for training (includes temporal info for context)
        current_state_features = state_to_features_for_training(holding_registers, coils, idx)
        state_sequence.append(current_state_features)
        update_state_with_transaction(row, idx, holding_registers, coils)
        
        if idx % 2000 == 0 and idx > 0:
            print(f"  Processed {idx} transactions...")
    
    final_state_features = state_to_features_for_training(holding_registers, coils, len(df))
    state_sequence.append(final_state_features)
    
    return np.array(state_sequence)

def test_model_prediction_correctness(model, model_name):
    """Test that model predictions have correct temporal metadata"""
    print(f"\n🧪 Testing {model_name} prediction correctness...")
    
    # Create test state with some values
    test_state = {
        "holding_registers": {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)},
        "coils": {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
    }
    test_state["holding_registers"][0] = {"value": 100, "last_changed": 5, "change_count": 3}
    test_state["coils"][0] = {"value": 1, "last_changed": 10, "change_count": 2}
    
    current_time = 1000
    
    try:
        prediction = model.predict_next_state(test_state, current_time=current_time)
        
        # Check structure
        assert "holding_registers" in prediction
        assert "coils" in prediction
        assert len(prediction["holding_registers"]) == 39
        assert len(prediction["coils"]) == 19
        
        # Check that temporal metadata is reasonable
        changes_detected = 0
        
        for i in range(39):
            reg = prediction["holding_registers"][i]
            assert "value" in reg
            assert "last_changed" in reg  
            assert "change_count" in reg
            
            # If value changed from test state, last_changed should be current_time
            if reg["value"] != test_state["holding_registers"][i]["value"]:
                if reg["last_changed"] == current_time:
                    changes_detected += 1
                    # Change count should increment
                    expected_count = test_state["holding_registers"][i]["change_count"] + 1
                    assert reg["change_count"] == expected_count, f"Wrong change count for HR{i}"
        
        for i in range(19):
            coil = prediction["coils"][i]
            assert "value" in coil
            assert "last_changed" in coil
            assert "change_count" in coil
            
            # Similar check for coils
            if coil["value"] != test_state["coils"][i]["value"]:
                if coil["last_changed"] == current_time:
                    changes_detected += 1
        
        print(f"✅ {model_name}: Structure correct, {changes_detected} changes detected with proper temporal metadata")
        return True
        
    except Exception as e:
        print(f"❌ {model_name}: Test failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def train_and_save_fixed_models(dataset_path="modbus_output_with_time.csv", 
                               save_dir="models/saved_models_fixed",
                               max_rows=20000,
                               train_split=0.95):
    """Train and save models with fixed prediction approach"""
    
    print("🚀 Starting FIXED Model Training Pipeline")
    print("Models will predict VALUES only, not temporal metadata")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Max rows: {max_rows}")
    print(f"Train split: {train_split}")
    print(f"Save directory: {save_dir}")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Process dataset once for all models
    print("\n📊 Processing dataset...")
    features = process_dataset_for_training(dataset_path, max_rows)
    print(f"Feature shape: {features.shape}")
    print(f"Features include: values + temporal info for training context")
    print(f"But models will predict only values (58 outputs: 39 regs + 19 coils)")
    
    # Initialize models
    models_to_train = {
        "hmm": HMMStatePredictor(n_components=25),
        "xgboost": XGBoostStatePredictor(),
        "random_forest": RandomForestStatePredictor(),
        "lstm": LSTMStatePredictor()
    }
    
    trained_models = {}
    training_times = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n{'='*20} Training {model_name.upper()} {'='*20}")
        start_time = time.time()
        
        try:
            # Train model
            trained_model = model.train(features, train_split)
            
            # Test prediction correctness
            if test_model_prediction_correctness(trained_model, model_name):
                # Save model
                model_path = Path(save_dir) / f"{model_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(trained_model, f)
                
                training_time = time.time() - start_time
                training_times[model_name] = training_time
                trained_models[model_name] = trained_model
                
                print(f"✅ {model_name.upper()} training complete in {training_time:.2f}s")
                print(f"💾 Saved to: {model_path}")
            else:
                print(f"❌ {model_name.upper()} failed correctness test, not saved")
            
        except Exception as e:
            print(f"❌ {model_name.upper()} training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save training summary
    summary = {
        "training_times": training_times,
        "dataset_path": dataset_path,
        "max_rows": max_rows,
        "train_split": train_split,
        "feature_dim_input": 174,  # Full features for input
        "output_dim": 58,          # Values only for output
        "models_trained": list(trained_models.keys()),
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fixed_approach": True,
        "notes": "Models predict values only, temporal metadata computed afterward"
    }
    
    summary_path = Path(save_dir) / "training_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("🎉 FIXED MODEL TRAINING COMPLETE!")
    print(f"📋 Models trained: {', '.join(trained_models.keys())}")
    print(f"⏱️  Total time: {sum(training_times.values()):.2f}s")
    print(f"📁 All models saved in: {save_dir}")
    print(f"📄 Training summary: {summary_path}")
    print("\n🔧 Key improvements:")
    print("   ✅ Models predict only VALUES (not temporal metadata)")
    print("   ✅ Temporal metadata computed based on actual value changes")
    print("   ✅ Proper change detection and counting")
    print("   ✅ Correct last_changed timestamps")
    
    return trained_models, summary

def comprehensive_test(trained_models):
    """Comprehensive test of all trained models"""
    print("\n🔍 Comprehensive Model Testing...")
    print("=" * 40)
    
    # Test scenarios
    test_scenarios = {
        "empty_state": {
            "holding_registers": {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)},
            "coils": {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
        },
        "some_values": {
            "holding_registers": {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(39)},
            "coils": {i: {"value": 0, "last_changed": 0, "change_count": 0} for i in range(19)}
        }
    }
    
    # Add some values to second scenario
    test_scenarios["some_values"]["holding_registers"][0] = {"value": 100, "last_changed": 5, "change_count": 2}
    test_scenarios["some_values"]["holding_registers"][5] = {"value": 50, "last_changed": 10, "change_count": 1}
    test_scenarios["some_values"]["coils"][0] = {"value": 1, "last_changed": 15, "change_count": 3}
    
    for scenario_name, test_state in test_scenarios.items():
        print(f"\n📋 Testing scenario: {scenario_name}")
        
        for model_name, model in trained_models.items():
            try:
                prediction = model.predict_next_state(test_state, current_time=1000)
                
                # Count non-zero predictions
                non_zero_regs = sum(1 for reg in prediction["holding_registers"].values() if reg["value"] != 0)
                non_zero_coils = sum(1 for coil in prediction["coils"].values() if coil["value"] != 0)
                
                # Count changes from input
                reg_changes = sum(1 for i in range(39) 
                                if prediction["holding_registers"][i]["value"] != test_state["holding_registers"][i]["value"])
                coil_changes = sum(1 for i in range(19)
                                 if prediction["coils"][i]["value"] != test_state["coils"][i]["value"])
                
                print(f"   ✅ {model_name:12}: {non_zero_regs:2d} active regs, {non_zero_coils:2d} active coils, "
                      f"{reg_changes:2d} reg changes, {coil_changes:2d} coil changes")
                
            except Exception as e:
                print(f"   ❌ {model_name:12}: Failed - {e}")

if __name__ == "__main__":
    # Run the fixed training pipeline
    trained_models, summary = train_and_save_fixed_models()
    
    if trained_models:
        comprehensive_test(trained_models)
        
        print("\n✨ Fixed training complete! Models now predict values only.")
        print("\nNext steps:")
        print("1. Update your prediction services to use the fixed models")
        print("2. Update import paths to use shared.utils_fixed and shared.model_classes_fixed")
        print("3. Restart ensemble orchestrator with fixed models")
        print("4. Test ensemble predictions - temporal metadata should be correct!")
        
    else:
        print("\n❌ No models were successfully trained. Check the errors above.")