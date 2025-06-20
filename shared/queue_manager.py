# shared/queue_manager.py
"""
Queue Manager for Individual Model Queues
Handles atomic request distribution and queue file operations
"""

import json
import time
import uuid
import threading
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import copy

class QueueManager:
    """Manages individual queue files for each model with atomic operations"""
    
    def __init__(self, queue_dir=".", model_names=None):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        
        # Default model names
        if model_names is None:
            model_names = ["llm", "xgboost", "hmm", "lstm", "random_forest"]
        
        self.model_names = model_names
        self.queue_files = {
            model: self.queue_dir / f"{model}_queue.json" 
            for model in model_names
        }
        
        # Individual locks for each queue file to minimize contention
        self.queue_locks = {
            model: threading.Lock() 
            for model in model_names
        }
        
        # Initialize queue files if they don't exist
        self._initialize_queue_files()
    
    def _initialize_queue_files(self):
        """Initialize empty queue files for all models"""
        for model, queue_file in self.queue_files.items():
            if not queue_file.exists():
                with self.queue_locks[model]:
                    initial_data = {
                        "pending_requests": [],
                        "metadata": {
                            "model_name": model,
                            "created_at": time.time(),
                            "total_requests_processed": 0,
                            "last_cleanup": time.time()
                        }
                    }
                    self._write_queue_file(queue_file, initial_data)
    
    def _write_queue_file(self, file_path: Path, data: Dict):
        """Atomic write to queue file using temp file + rename"""
        temp_file = None
        try:
            # Create temp file in same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp', 
                prefix=f'{file_path.stem}_',
                dir=file_path.parent
            )
            temp_file = Path(temp_path)
            
            # Write to temp file
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(file_path)
            
        except Exception as e:
            # Cleanup temp file if it exists
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            raise e
    
    def _read_queue_file(self, file_path: Path) -> Dict:
        """Read queue file with error handling"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                # Return empty structure if file doesn't exist
                return {
                    "pending_requests": [],
                    "metadata": {
                        "model_name": file_path.stem.replace('_queue', ''),
                        "created_at": time.time(),
                        "total_requests_processed": 0,
                        "last_cleanup": time.time()
                    }
                }
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading queue file {file_path}: {e}")
            # Return empty structure on error
            return {"pending_requests": [], "metadata": {}}
    
    def distribute_request_atomically(self, request_data: Dict, timeout: float = 20.0) -> str:
        """
        Atomically distribute the same request to all model queues
        
        Args:
            request_data: The request data to distribute
            timeout: Request timeout in seconds
            
        Returns:
            request_id: Unique identifier for this request
            
        Raises:
            Exception: If atomic distribution fails
        """
        request_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Prepare request for all queues
        request_item = {
            "request_id": request_id,
            "current_state": request_data,
            "timeout": timeout,
            "created_at": current_time,
            "expires_at": current_time + timeout
        }
        
        print(f"📤 Distributing request {request_id[:8]} to {len(self.model_names)} model queues")
        
        # Step 1: Acquire all locks (to prevent deadlock, acquire in consistent order)
        acquired_locks = []
        try:
            for model in sorted(self.model_names):  # Consistent ordering
                self.queue_locks[model].acquire()
                acquired_locks.append(model)
            
            # Step 2: Read all queue files
            queue_data = {}
            for model in self.model_names:
                queue_data[model] = self._read_queue_file(self.queue_files[model])
            
            # Step 3: Add request to all queues
            for model in self.model_names:
                queue_data[model]["pending_requests"].append(request_item)
                queue_data[model]["metadata"]["total_requests_processed"] = (
                    queue_data[model]["metadata"].get("total_requests_processed", 0) + 1
                )
            
            # Step 4: Write all queue files atomically
            temp_files = []
            try:
                # Create all temp files first
                for model in self.model_names:
                    temp_fd, temp_path = tempfile.mkstemp(
                        suffix='.tmp',
                        prefix=f'{model}_queue_',
                        dir=self.queue_files[model].parent
                    )
                    temp_file = Path(temp_path)
                    temp_files.append((temp_file, self.queue_files[model]))
                    
                    # Write to temp file
                    with os.fdopen(temp_fd, 'w') as f:
                        json.dump(queue_data[model], f, indent=2)
                
                # Atomic rename all files
                for temp_file, target_file in temp_files:
                    temp_file.rename(target_file)
                
                print(f"✅ Request {request_id[:8]} distributed to all {len(self.model_names)} queues")
                return request_id
                
            except Exception as e:
                # Rollback: cleanup any temp files
                for temp_file, _ in temp_files:
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except:
                            pass
                raise e
            
        finally:
            # Release all locks in reverse order
            for model in reversed(acquired_locks):
                self.queue_locks[model].release()
    
    def get_pending_requests(self, model_name: str) -> List[Dict]:
        """
        Get pending requests for a specific model (simplified - no filtering needed)
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of pending request dictionaries
        """
        if model_name not in self.queue_files:
            return []
        
        with self.queue_locks[model_name]:
            queue_data = self._read_queue_file(self.queue_files[model_name])
            current_time = time.time()
            
            # Filter out expired requests
            valid_requests = []
            expired_requests = []
            
            for request in queue_data["pending_requests"]:
                if current_time <= request.get("expires_at", float('inf')):
                    # Add remaining time for compatibility
                    request["remaining_time"] = request.get("expires_at", current_time) - current_time
                    valid_requests.append(request)
                else:
                    expired_requests.append(request)
            
            # Remove expired requests if any found
            if expired_requests:
                queue_data["pending_requests"] = valid_requests
                queue_data["metadata"]["last_cleanup"] = current_time
                self._write_queue_file(self.queue_files[model_name], queue_data)
                print(f"🧹 Cleaned {len(expired_requests)} expired requests from {model_name} queue")
            
            return valid_requests
    
    def remove_request_from_queue(self, model_name: str, request_id: str) -> bool:
        """
        Remove a specific request from a model's queue after processing
        
        Args:
            model_name: Name of the model
            request_id: ID of the request to remove
            
        Returns:
            bool: True if request was found and removed
        """
        if model_name not in self.queue_files:
            return False
        
        with self.queue_locks[model_name]:
            queue_data = self._read_queue_file(self.queue_files[model_name])
            
            # Find and remove the request
            original_count = len(queue_data["pending_requests"])
            queue_data["pending_requests"] = [
                req for req in queue_data["pending_requests"] 
                if req.get("request_id") != request_id
            ]
            
            removed = len(queue_data["pending_requests"]) < original_count
            
            if removed:
                # Update metadata
                queue_data["metadata"]["last_cleanup"] = time.time()
                self._write_queue_file(self.queue_files[model_name], queue_data)
                print(f"✅ Removed request {request_id[:8]} from {model_name} queue")
            
            return removed
    
    def get_queue_status(self, model_name: str = None) -> Dict:
        """
        Get status information for queues
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Dictionary with queue status information
        """
        if model_name:
            models_to_check = [model_name] if model_name in self.queue_files else []
        else:
            models_to_check = self.model_names
        
        status = {}
        current_time = time.time()
        
        for model in models_to_check:
            with self.queue_locks[model]:
                queue_data = self._read_queue_file(self.queue_files[model])
                
                pending_count = len(queue_data["pending_requests"])
                expired_count = sum(
                    1 for req in queue_data["pending_requests"]
                    if current_time > req.get("expires_at", float('inf'))
                )
                
                status[model] = {
                    "pending_requests": pending_count,
                    "expired_requests": expired_count,
                    "valid_requests": pending_count - expired_count,
                    "total_processed": queue_data["metadata"].get("total_requests_processed", 0),
                    "last_cleanup": queue_data["metadata"].get("last_cleanup", 0),
                    "queue_file": str(self.queue_files[model])
                }
        
        return status
    
    def cleanup_all_queues(self, max_age_seconds: float = 300.0) -> Dict[str, int]:
        """
        Clean up expired/old requests from all queues
        
        Args:
            max_age_seconds: Maximum age for requests before cleanup
            
        Returns:
            Dictionary mapping model names to number of requests cleaned
        """
        cleanup_results = {}
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        for model in self.model_names:
            with self.queue_locks[model]:
                queue_data = self._read_queue_file(self.queue_files[model])
                
                original_count = len(queue_data["pending_requests"])
                
                # Keep only non-expired and recent requests
                queue_data["pending_requests"] = [
                    req for req in queue_data["pending_requests"]
                    if (req.get("expires_at", 0) > current_time and 
                        req.get("created_at", 0) > cutoff_time)
                ]
                
                cleaned_count = original_count - len(queue_data["pending_requests"])
                cleanup_results[model] = cleaned_count
                
                if cleaned_count > 0:
                    queue_data["metadata"]["last_cleanup"] = current_time
                    self._write_queue_file(self.queue_files[model], queue_data)
        
        total_cleaned = sum(cleanup_results.values())
        if total_cleaned > 0:
            print(f"🧹 Cleaned {total_cleaned} old requests across all queues")
        
        return cleanup_results
    
    def get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths for all models"""
        depths = {}
        for model in self.model_names:
            with self.queue_locks[model]:
                queue_data = self._read_queue_file(self.queue_files[model])
                depths[model] = len(queue_data["pending_requests"])
        return depths
    
    def emergency_clear_queue(self, model_name: str) -> int:
        """
        Emergency clear of a specific model's queue
        
        Args:
            model_name: Model whose queue should be cleared
            
        Returns:
            Number of requests that were cleared
        """
        if model_name not in self.queue_files:
            return 0
        
        with self.queue_locks[model_name]:
            queue_data = self._read_queue_file(self.queue_files[model_name])
            cleared_count = len(queue_data["pending_requests"])
            
            queue_data["pending_requests"] = []
            queue_data["metadata"]["last_cleanup"] = time.time()
            
            self._write_queue_file(self.queue_files[model_name], queue_data)
            print(f"⚠️  Emergency cleared {cleared_count} requests from {model_name} queue")
            
            return cleared_count