# restart_services.py
"""
Script to cleanly restart all ensemble services
"""

import subprocess
import time
import os
import signal
import psutil

def find_ensemble_processes():
    """Find all running ensemble processes"""
    ensemble_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            if any(script in cmdline for script in [
                'orchestrator.py',
                'hmm_predictor.py', 
                'llm_predictor.py',
                'ml_predictors.py'
            ]):
                ensemble_processes.append(proc)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return ensemble_processes

def stop_ensemble_services():
    """Stop all running ensemble services"""
    print("🛑 Stopping ensemble services...")
    
    processes = find_ensemble_processes()
    
    if not processes:
        print("   No ensemble processes found")
        return
    
    for proc in processes:
        try:
            cmdline = ' '.join(proc.cmdline())
            print(f"   Stopping: {cmdline}")
            proc.terminate()
        except Exception as e:
            print(f"   Error stopping process: {e}")
    
    # Wait for processes to terminate
    print("   Waiting for processes to stop...")
    time.sleep(3)
    
    # Force kill any remaining processes
    remaining = find_ensemble_processes()
    if remaining:
        print("   Force killing remaining processes...")
        for proc in remaining:
            try:
                proc.kill()
            except:
                pass

def start_ensemble_services():
    """Start all ensemble services"""
    print("🚀 Starting ensemble services...")
    
    services = [
        ("HMM Service", ["python", "models/hmm_predictor.py"]),
        ("XGBoost Service", ["python", "models/ml_predictors.py", "xgboost"]),
        ("Random Forest Service", ["python", "models/ml_predictors.py", "random_forest"]),
        ("LSTM Service", ["python", "models/ml_predictors.py", "lstm"]),
        ("LLM Service", ["python", "models/llm_predictor.py"]),
    ]
    
    started_services = []
    
    for service_name, cmd in services:
        try:
            print(f"   Starting {service_name}...")
            
            # Start service in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
            
            started_services.append((service_name, process))
            time.sleep(2)  # Stagger startup
            
        except Exception as e:
            print(f"   ❌ Failed to start {service_name}: {e}")
    
    print(f"✅ Started {len(started_services)} services")
    
    # Wait for services to initialize
    print("   Waiting for services to initialize...")
    time.sleep(5)
    
    return started_services

def check_services_status():
    """Check status of ensemble services"""
    print("📊 Checking service status...")
    
    from shared.state_store import StateStore
    store = StateStore()
    
    try:
        active_models = store.get_active_models()
        print(f"   Active models: {active_models}")
        
        if len(active_models) >= 3:
            print("✅ Ensemble system is ready!")
            return True
        else:
            print("⚠️  Not all services are ready yet")
            return False
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

def main():
    """Main restart function"""
    print("🔄 Ensemble Service Restart Script")
    print("=" * 40)
    
    # Stop existing services
    stop_ensemble_services()
    
    # Clean up shared state
    try:
        if os.path.exists("shared_state.json"):
            os.remove("shared_state.json")
            print("🧹 Cleaned up shared state")
    except:
        pass
    
    # Start services
    started_services = start_ensemble_services()
    
    # Check status
    time.sleep(3)
    ready = check_services_status()
    
    if ready:
        print("\n🎉 Ensemble system restarted successfully!")
        print("🎯 Next step: Run 'python orchestrator.py' in a new terminal")
    else:
        print("\n⚠️  Some services may not be ready yet")
        print("   Wait a few more seconds and check again")
    
    print(f"\n📋 Started services:")
    for service_name, process in started_services:
        status = "Running" if process.poll() is None else "Stopped"
        print(f"   {service_name}: {status} (PID: {process.pid})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Restart interrupted by user")
    except Exception as e:
        print(f"❌ Restart failed: {e}")