# gpu_diagnostic.py
"""
Diagnostic script to check GPU setup and memory for LLM loading
"""

import torch
import psutil
import os

def check_gpu_setup():
    """Check GPU availability and memory"""
    print("🔍 GPU DIAGNOSTIC REPORT")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total Memory: {memory_gb:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Check current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Currently Allocated: {allocated:.1f} GB")
                print(f"  Currently Reserved: {reserved:.1f} GB")
                print(f"  Available: {memory_gb - reserved:.1f} GB")
    else:
        print("❌ No CUDA GPUs detected")
    
    # Check system RAM
    ram_gb = psutil.virtual_memory().total / 1024**3
    ram_available = psutil.virtual_memory().available / 1024**3
    print(f"\n💾 System RAM: {ram_gb:.1f} GB")
    print(f"Available RAM: {ram_available:.1f} GB")
    
    # Check environment variables
    print(f"\n🌍 Environment Variables:")
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 8:
            print("✅ Sufficient GPU memory for 4-bit quantization")
            print("   Recommended: Use GPU with load_in_4bit=True")
        elif gpu_memory >= 4:
            print("⚠️  Limited GPU memory")
            print("   Recommended: Use GPU with CPU offloading")
            print("   Set: llm_int8_enable_fp32_cpu_offload=True")
        else:
            print("❌ Very limited GPU memory")
            print("   Recommended: Use CPU or hybrid loading")
    else:
        print("📱 No GPU detected")
        print("   Recommended: Use CPU-only loading")
    
    return torch.cuda.is_available()

def test_model_loading_strategies():
    """Test different model loading strategies"""
    print(f"\n🧪 TESTING MODEL LOADING STRATEGIES")
    print("=" * 50)
    
    strategies = [
        {
            "name": "GPU 4-bit",
            "params": {
                "load_in_4bit": True,
                "device_map": "auto",
                "llm_int8_enable_fp32_cpu_offload": False
            }
        },
        {
            "name": "GPU 4-bit + CPU offload",
            "params": {
                "load_in_4bit": True,
                "device_map": "auto", 
                "llm_int8_enable_fp32_cpu_offload": True
            }
        },
        {
            "name": "CPU only",
            "params": {
                "load_in_4bit": False,
                "device_map": "cpu",
                "llm_int8_enable_fp32_cpu_offload": False
            }
        }
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        print(f"  Parameters: {strategy['params']}")
        
        # Estimate memory requirements
        if strategy['params']['load_in_4bit']:
            print("  Estimated GPU memory needed: 4-6 GB")
        else:
            print("  Estimated memory needed: 8-12 GB")

if __name__ == "__main__":
    has_gpu = check_gpu_setup()
    test_model_loading_strategies()
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Update your LLM predictor with the fixed loading code")
    print("2. Use the recommended strategy based on your GPU memory")
    print("3. If loading still fails, try the CPU fallback")
    print("4. Monitor GPU memory usage with: watch -n 1 nvidia-smi")