"""
GPU management utilities for monitoring and setup.

This module provides centralized GPU management including:
- Device detection and setup
- Memory monitoring
- Background monitoring thread
- GPU cleanup utilities
"""

import subprocess
import threading
import time
from typing import Optional, Dict, Any

import torch
from IPython.display import clear_output


class GPUManager:
    """
    Centralized GPU management for PyTorch training.
    
    Handles device detection, memory monitoring, and cleanup operations.
    """
    
    def __init__(self):
        """Initialize GPU manager."""
        self.device = None
        self.gpu_available = False
        self.gpu_name = None
        self.total_memory_gb = 0.0
        self.monitor_thread = None
        self._monitoring = False
        
    def setup(self) -> torch.device:
        """
        Setup and detect GPU device.
        
        Returns:
            torch.device: CUDA device if available, else CPU
        """
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.device = torch.device('cuda')
            self.gpu_name = torch.cuda.get_device_name(0)
            self.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            self.device = torch.device('cpu')
            
        return self.device
    
    def print_info(self):
        """Print GPU information."""
        print(f"CUDA available: {self.gpu_available}")
        if self.gpu_available:
            print(f"CUDA device: {self.gpu_name}")
            print(f"CUDA memory: {self.total_memory_gb:.2f} GB")
        else:
            print("Running on CPU")
    
    def get_nvidia_smi_info(self) -> Optional[list]:
        """
        Get GPU information from nvidia-smi.
        
        Returns:
            List of GPU info strings, or None if unavailable
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, 
                text=True, 
                check=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except:
            pass
        return None
    
    def print_nvidia_smi_info(self):
        """Print detailed GPU information from nvidia-smi."""
        gpu_info = self.get_nvidia_smi_info()
        if gpu_info:
            for i, gpu in enumerate(gpu_info):
                name, memory = gpu.split(', ')
                print(f"GPU {i}: {name}, {memory} MB VRAM")
        else:
            print("GPU: Not detected or nvidia-smi unavailable")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current GPU memory statistics.
        
        Returns:
            Dictionary with memory stats in GB
        """
        if not self.gpu_available:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'free_gb': 0.0,
                'total_gb': 0.0
            }
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = self.total_memory_gb
        free = total - allocated
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_gb': total
        }
    
    def print_memory_stats(self):
        """Print current GPU memory usage."""
        stats = self.get_memory_stats()
        
        if self.gpu_available:
            print(f"\nGPU Memory Usage:")
            print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
            print(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
            print(f"  Free:      {stats['free_gb']:.2f} GB")
            print(f"  Total:     {stats['total_gb']:.2f} GB")
        else:
            print("GPU not available - running on CPU")
    
    def cleanup(self):
        """Clear GPU cache and collect garbage."""
        if self.gpu_available:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    def monitor_memory(
        self, 
        duration: int = 3600, 
        interval: int = 30,
        stop_on_error: bool = True
    ):
        """
        Monitor GPU memory usage in background thread.
        
        Args:
            duration: Total monitoring duration in seconds (default 1 hour)
            interval: Update interval in seconds (default 10s)
            stop_on_error: Stop monitoring after consecutive errors
        """
        if not self.gpu_available:
            print("GPU not available. Memory monitoring disabled.")
            return
        
        def monitor():
            error_count = 0
            max_errors = 3
            start_time = time.time()
            clear_enabled = True
            
            print(f"Starting GPU memory monitoring (interval: {interval}s, duration: {duration}s)")
            print("Press Ctrl+C to stop monitoring\n")
            
            while self._monitoring and (time.time() - start_time) < duration:
                try:
                    if clear_enabled:
                        clear_output(wait=True)
                    
                    stats = self.get_memory_stats()
                    elapsed = time.time() - start_time
                    
                    print(f"GPU Memory Monitor - Elapsed: {elapsed:.0f}s / {duration}s")
                    print("="*60)
                    print(f"Device: {self.gpu_name}")
                    print(f"Allocated: {stats['allocated_gb']:.2f} GB")
                    print(f"Reserved:  {stats['reserved_gb']:.2f} GB")
                    print(f"Free:      {stats['free_gb']:.2f} GB")
                    print(f"Total:     {stats['total_gb']:.2f} GB")
                    print(f"Usage:     {(stats['allocated_gb']/stats['total_gb'])*100:.1f}%")
                    print("="*60)
                    print(f"Next update in {interval} seconds... (Ctrl+C to stop)")
                    
                    error_count = 0  # Reset error count on success
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    print("\n\nMonitoring stopped by user.")
                    break
                except Exception as e:
                    error_count += 1
                    print(f"\nError reading GPU stats: {e}")
                    print(f"Error count: {error_count}/{max_errors}")
                    
                    if stop_on_error and error_count >= max_errors:
                        clear_enabled = False
                        print(f"\nStopping monitor after {max_errors} consecutive errors.")
                        break
                    
                    if error_count >= max_errors:
                        clear_enabled = False
                    
                    time.sleep(interval)
            
            if self._monitoring:
                print(f"\nMonitoring completed after {time.time() - start_time:.0f}s")
            
            self._monitoring = False
        
        self._monitoring = True
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
        return self.monitor_thread
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def recommend_batch_size(self, model_size_gb: float = 2.0) -> int:
        """
        Recommend batch size based on available memory.
        
        Args:
            model_size_gb: Estimated model size in GB
            
        Returns:
            Recommended batch size
        """
        if not self.gpu_available:
            return 4  # Conservative for CPU
        
        stats = self.get_memory_stats()
        available_gb = stats['free_gb']
        
        # Conservative estimate: leave 2GB buffer
        usable_gb = max(0, available_gb - 2.0)
        
        # Rough estimate: each sample uses ~0.1GB with augmentation
        samples_per_gb = 10
        max_batch = int(usable_gb * samples_per_gb)
        
        # Clamp to reasonable range
        if max_batch < 4:
            return 4
        elif max_batch > 64:
            return 64
        else:
            # Round to nearest power of 2
            import math
            power = int(math.log2(max_batch))
            return 2 ** power
    
    def get_device(self) -> torch.device:
        """
        Get the configured device.
        
        Returns:
            torch.device: Configured device
        """
        if self.device is None:
            self.setup()
        return self.device
    
    def is_available(self) -> bool:
        """
        Check if GPU is available.
        
        Returns:
            bool: True if GPU available
        """
        return self.gpu_available


# Global instance for convenience
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """
    Get global GPU manager instance.
    
    Returns:
        GPUManager: Global GPU manager
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
        _gpu_manager.setup()
    return _gpu_manager
