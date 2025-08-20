"""
Safety utilities for robust CBM training
"""

import torch
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable, Generator
from pathlib import Path
import json
import psutil
import gc

from .logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)


@contextmanager
def safe_training_context(operation_name: str, 
                         cleanup_fn: Optional[Callable] = None,
                         save_on_error: bool = True,
                         error_save_path: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """
    Enhanced context manager for safe training operations with error handling and cleanup
    
    Args:
        operation_name: Name of the operation for logging
        cleanup_fn: Optional cleanup function to call on error
        save_on_error: Whether to save state on error
        error_save_path: Path to save error state
    
    Yields:
        Dictionary with operation context information
    """
    start_time = time.time()
    context = {
        'operation': operation_name,
        'start_time': start_time,
        'success': False,
        'error': None,
        'duration': 0
    }
    
    logger.info(f"🚀 Starting {operation_name}")
    
    try:
        yield context
        
        # Mark as successful
        context['success'] = True
        context['duration'] = time.time() - start_time
        
        logger.info(f"✅ {operation_name} completed successfully in {context['duration']:.2f}s")
        
    except KeyboardInterrupt:
        context['error'] = "Interrupted by user"
        context['duration'] = time.time() - start_time
        
        logger.warning(f"⏹️ {operation_name} interrupted by user after {context['duration']:.2f}s")
        
        if cleanup_fn:
            logger.info("🧹 Running cleanup function...")
            try:
                cleanup_fn()
            except Exception as cleanup_error:
                logger.error(f"❌ Cleanup failed: {cleanup_error}")
        
        raise
        
    except Exception as e:
        context['error'] = str(e)
        context['duration'] = time.time() - start_time
        
        logger.error(f"❌ {operation_name} failed after {context['duration']:.2f}s: {str(e)}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        # Save error state if requested
        if save_on_error:
            try:
                _save_error_state(operation_name, e, error_save_path)
            except Exception as save_error:
                logger.error(f"❌ Failed to save error state: {save_error}")
        
        # Run cleanup function
        if cleanup_fn:
            logger.info("🧹 Running cleanup function...")
            try:
                cleanup_fn()
            except Exception as cleanup_error:
                logger.error(f"❌ Cleanup failed: {cleanup_error}")
        
        raise


def _save_error_state(operation_name: str, error: Exception, save_path: Optional[str] = None):
    """Save error state for debugging"""
    if save_path is None:
        save_path = f"error_states/{operation_name}_{int(time.time())}"
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    error_info = {
        'operation': operation_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'timestamp': time.time(),
        'system_info': get_system_info()
    }
    
    with open(save_path / "error_info.json", 'w') as f:
        json.dump(error_info, f, indent=2, default=str)
    
    logger.info(f"💾 Error state saved to {save_path}")


@contextmanager
def memory_management_context(clear_cache_before: bool = True,
                            clear_cache_after: bool = True,
                            log_memory_usage: bool = True) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for memory management during training
    
    Args:
        clear_cache_before: Clear CUDA cache before operation
        clear_cache_after: Clear CUDA cache after operation  
        log_memory_usage: Log memory usage statistics
    """
    context = {}
    
    # Initial memory state
    if log_memory_usage:
        initial_memory = get_memory_info()
        context['initial_memory'] = initial_memory
        logger.debug(f"📊 Initial memory: {initial_memory}")
    
    # Clear cache before
    if clear_cache_before and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("🧹 Cleared CUDA cache before operation")
    
    try:
        yield context
        
    finally:
        # Clear cache after
        if clear_cache_after and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("🧹 Cleared CUDA cache after operation")
        
        # Final memory state
        if log_memory_usage:
            final_memory = get_memory_info()
            context['final_memory'] = final_memory
            
            if 'initial_memory' in context:
                memory_diff = {}
                for key in initial_memory:
                    if key in final_memory:
                        memory_diff[f"{key}_change"] = final_memory[key] - initial_memory[key]
                context['memory_diff'] = memory_diff
                
                logger.debug(f"📊 Memory change: {memory_diff}")


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage information"""
    info = {}
    
    # System memory
    memory = psutil.virtual_memory()
    info['system_memory_used_gb'] = memory.used / (1024**3)
    info['system_memory_percent'] = memory.percent
    
    # CUDA memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_memory = torch.cuda.memory_stats(i)
            info[f'cuda_{i}_allocated_gb'] = device_memory.get('allocated_bytes.all.current', 0) / (1024**3)
            info[f'cuda_{i}_reserved_gb'] = device_memory.get('reserved_bytes.all.current', 0) / (1024**3)
    
    return info


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for debugging"""
    info = {
        'python_version': psutil.sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info


class SafetyMonitor:
    """Monitor training safety metrics and trigger interventions"""
    
    def __init__(self, 
                 memory_threshold: float = 0.9,
                 loss_explosion_threshold: float = 1e6,
                 gradient_norm_threshold: float = 1e3,
                 check_frequency: int = 10):
        self.memory_threshold = memory_threshold
        self.loss_explosion_threshold = loss_explosion_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        self.check_frequency = check_frequency
        
        self.step_counter = 0
        self.warnings_issued = set()
        
        self.logger = setup_enhanced_logging(f"{__name__}.SafetyMonitor")
    
    def check_safety(self, loss: float, model: torch.nn.Module = None) -> Dict[str, Any]:
        """
        Check various safety metrics and return status
        
        Args:
            loss: Current loss value
            model: Model to check gradients
        
        Returns:
            Dictionary with safety status and recommendations
        """
        self.step_counter += 1
        safety_status = {
            'safe': True,
            'warnings': [],
            'critical_issues': [],
            'recommendations': []
        }
        
        # Only perform checks at specified frequency
        if self.step_counter % self.check_frequency != 0:
            return safety_status
        
        # Check memory usage
        memory_info = get_memory_info()
        if memory_info.get('system_memory_percent', 0) > self.memory_threshold * 100:
            warning = f"High memory usage: {memory_info['system_memory_percent']:.1f}%"
            if warning not in self.warnings_issued:
                safety_status['warnings'].append(warning)
                safety_status['recommendations'].append("Consider reducing batch size or clearing unnecessary variables")
                self.warnings_issued.add(warning)
                self.logger.warning(f"⚠️ {warning}")
        
        # Check CUDA memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated_gb = memory_info.get(f'cuda_{i}_allocated_gb', 0)
                if allocated_gb > 10:  # Arbitrary threshold for large models
                    warning = f"High CUDA memory on device {i}: {allocated_gb:.1f}GB"
                    if warning not in self.warnings_issued:
                        safety_status['warnings'].append(warning)
                        self.warnings_issued.add(warning)
        
        # Check loss explosion
        if abs(loss) > self.loss_explosion_threshold:
            issue = f"Loss explosion detected: {loss:.2e}"
            safety_status['critical_issues'].append(issue)
            safety_status['safe'] = False
            safety_status['recommendations'].append("Reduce learning rate or check for numerical instabilities")
            self.logger.error(f"❌ {issue}")
        
        # Check for NaN loss
        if torch.isnan(torch.tensor(loss)):
            issue = "NaN loss detected"
            safety_status['critical_issues'].append(issue)
            safety_status['safe'] = False
            safety_status['recommendations'].append("Check for division by zero or invalid operations")
            self.logger.error(f"❌ {issue}")
        
        # Check gradient norms if model provided
        if model is not None:
            try:
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    total_norm = total_norm ** (1. / 2)
                    
                    if total_norm > self.gradient_norm_threshold:
                        issue = f"Large gradient norm: {total_norm:.2e}"
                        safety_status['critical_issues'].append(issue)
                        safety_status['safe'] = False
                        safety_status['recommendations'].append("Apply gradient clipping or reduce learning rate")
                        self.logger.error(f"❌ {issue}")
                    
                    # Check for zero gradients
                    if total_norm < 1e-8:
                        warning = f"Very small gradient norm: {total_norm:.2e}"
                        if warning not in self.warnings_issued:
                            safety_status['warnings'].append(warning)
                            safety_status['recommendations'].append("Check if model is learning or if gradients are vanishing")
                            self.warnings_issued.add(warning)
            
            except Exception as e:
                self.logger.debug(f"Error checking gradients: {e}")
        
        return safety_status
    
    def reset_warnings(self):
        """Reset warning tracking"""
        self.warnings_issued.clear()
        self.step_counter = 0


class CheckpointManager:
    """Manage training checkpoints with safety features"""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 max_checkpoints: int = 5,
                 save_frequency: int = 100):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_frequency = save_frequency
        
        self.checkpoint_counter = 0
        self.logger = setup_enhanced_logging(f"{__name__}.CheckpointManager")
    
    def should_save_checkpoint(self, step: int) -> bool:
        """Check if checkpoint should be saved at current step"""
        return step % self.save_frequency == 0
    
    def save_checkpoint(self, 
                       step: int,
                       model_state: Dict[str, Any],
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, Any]] = None,
                       is_best: bool = False) -> str:
        """
        Save training checkpoint with safety checks
        
        Args:
            step: Current training step
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step}.pt"
        if is_best:
            checkpoint_name = f"best_checkpoint_step_{step}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            checkpoint_data = {
                'step': step,
                'model_state': model_state,
                'timestamp': time.time(),
                'system_info': get_system_info()
            }
            
            if optimizer_state:
                checkpoint_data['optimizer_state'] = optimizer_state
            
            if metrics:
                checkpoint_data['metrics'] = metrics
            
            # Save with temporary name first for safety
            temp_path = checkpoint_path.with_suffix('.tmp')
            torch.save(checkpoint_data, temp_path)
            
            # Rename to final name if save successful
            temp_path.rename(checkpoint_path)
            
            self.checkpoint_counter += 1
            self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints(keep_best=is_best)
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")
            raise
    
    def _cleanup_old_checkpoints(self, keep_best: bool = True):
        """Remove old checkpoints to save space"""
        try:
            # Get all checkpoint files
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
            best_checkpoints = list(self.checkpoint_dir.glob("best_checkpoint_step_*.pt"))
            
            # Sort by creation time
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove excess regular checkpoints
            if len(checkpoints) > self.max_checkpoints:
                to_remove = checkpoints[:-self.max_checkpoints]
                for checkpoint in to_remove:
                    checkpoint.unlink()
                    self.logger.debug(f"🗑️ Removed old checkpoint: {checkpoint}")
            
            # Keep only most recent best checkpoint
            if keep_best and len(best_checkpoints) > 1:
                best_checkpoints.sort(key=lambda x: x.stat().st_mtime)
                to_remove = best_checkpoints[:-1]
                for checkpoint in to_remove:
                    checkpoint.unlink()
                    self.logger.debug(f"🗑️ Removed old best checkpoint: {checkpoint}")
        
        except Exception as e:
            self.logger.warning(f"⚠️ Error cleaning up checkpoints: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint"""
        try:
            checkpoints = list(self.checkpoint_dir.glob("*.pt"))
            if not checkpoints:
                return None
            
            # Get most recent checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
            self.logger.info(f"📂 Loaded checkpoint: {latest_checkpoint}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            return None


# Utility functions for common safety patterns
def with_error_handling(func: Callable, 
                       operation_name: str = None,
                       default_return: Any = None,
                       reraise: bool = True) -> Callable:
    """Decorator for adding error handling to functions"""
    
    def wrapper(*args, **kwargs):
        op_name = operation_name or func.__name__
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ Error in {op_name}: {e}")
            
            if reraise:
                raise
            else:
                return default_return
    
    return wrapper


def safe_tensor_operation(operation: Callable, 
                         *tensors: torch.Tensor,
                         fallback_value: Any = None) -> Any:
    """Safely perform tensor operations with fallback"""
    try:
        # Check for invalid tensors
        for tensor in tensors:
            if torch.isnan(tensor).any():
                raise ValueError("NaN values detected in tensor")
            if torch.isinf(tensor).any():
                raise ValueError("Infinite values detected in tensor")
        
        return operation(*tensors)
        
    except Exception as e:
        logger.warning(f"⚠️ Tensor operation failed: {e}")
        if fallback_value is not None:
            return fallback_value
        raise