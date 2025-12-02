"""
Light training pipeline for quick model validation.

This module provides a lightweight training pipeline to quickly validate that models
can train successfully before committing to full training runs.
"""

import time
from typing import Dict, Any, Optional

import torch
import torch.optim as optim

from models import create_model
from losses import create_loss_function
from gpu_manager import GPUManager
import config


class LightPipeline:
    """
    Lightweight training pipeline for quick model validation.
    
    Runs abbreviated training (few epochs, limited batches) to verify:
    - Model architecture is valid
    - Forward/backward passes work
    - GPU memory is sufficient
    - Training loop executes without errors
    """
    
    def __init__(self, light_config: Dict[str, Any], class_weights: torch.Tensor):
        """
        Initialize light pipeline.
        
        Args:
            light_config: Configuration dictionary with training parameters
            class_weights: Class weights tensor for loss function
        """
        self.config = light_config
        self.class_weights = class_weights
        self.results = {}
        self.gpu_mgr = GPUManager()
        self.gpu_mgr.setup()
    
    def validate_model(
        self, 
        model_name: str, 
        train_loader, 
        val_loader
    ) -> Dict[str, Any]:
        """
        Quick validation that a model can train.
        
        Args:
            model_name: Name of the model to validate
            train_loader: Training data loader
            val_loader: Validation data loader (currently unused in light validation)
            
        Returns:
            Dictionary with validation results:
                - status: 'passed' or 'failed'
                - time: Time taken in seconds
                - error: Error message if failed, None otherwise
                - final_loss: Final average loss if passed
        """
        print(f"\n{'='*60}")
        print(f"Validating {model_name.upper()}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Aggressive GPU memory cleanup
            self.gpu_mgr.cleanup()
            
            # Create model
            model = self._create_model(model_name)
            
            # Create loss and optimizer
            loss_fn = self._create_loss_function()
            optimizer = self._create_optimizer(model)
            
            # Run quick training loop
            final_loss = self._training_loop(
                model, loss_fn, optimizer, train_loader, model_name
            )
            
            elapsed = time.time() - start_time
            print(f"  [PASSED] ({elapsed:.1f}s) - Final Loss: {final_loss:.4f}")
            
            # Cleanup
            self._cleanup_model(model, loss_fn, optimizer)
            
            return {
                'status': 'passed', 
                'time': elapsed, 
                'error': None,
                'final_loss': final_loss
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  [FAILED] ({elapsed:.1f}s)")
            print(f"  Error: {str(e)}")
            
            # Cleanup on error
            self.gpu_mgr.cleanup()
            
            return {
                'status': 'failed', 
                'time': elapsed, 
                'error': str(e),
                'final_loss': None
            }
    
    def validate_all_models(
        self, 
        model_names: list, 
        train_loader, 
        val_loader
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate all models in the list.
        
        Args:
            model_names: List of model names to validate
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary mapping model names to validation results
        """
        print("\n" + "="*80)
        print("LIGHT PIPELINE VALIDATION - Testing all models can train")
        print("="*80)
        print(f"Config: {self.config['num_epochs']} epochs, "
              f"{self.config.get('max_batches_per_epoch', 'all')} batches/epoch")
        print(f"Expected time: ~2-3 minutes per model, "
              f"~{len(model_names)*2.5:.0f} minutes total\n")
        
        self.results = {}
        
        for model_name in model_names:
            result = self.validate_model(model_name, train_loader, val_loader)
            self.results[model_name] = result
        
        # Print summary
        self._print_summary(model_names)
        
        return self.results
    
    def _create_model(self, model_name: str):
        """Create and initialize model."""
        model = create_model(
            model_name=model_name,
            in_channels=6 if 'siamese' not in model_name.lower() else 3,
            num_classes=config.NUM_CLASSES,
            **config.MODEL_CONFIGS.get(model_name, {})
        )
        return model.to(self.config['device'])
    
    def _create_loss_function(self):
        """Create loss function."""
        return create_loss_function(
            loss_type=self.config['loss_type'],
            num_classes=config.NUM_CLASSES,
            class_weights=self.class_weights.to(self.config['device']),
            device=self.config['device'],
            ce_weight=self.config.get('ce_weight', 0.1),
            dice_weight=self.config.get('dice_weight', 2.0),
            focal_weight=self.config.get('focal_weight', 3.0),
            focal_gamma=self.config.get('focal_gamma', 3.0)
        )
    
    def _create_optimizer(self, model):
        """Create optimizer."""
        return optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    def _training_loop(self, model, loss_fn, optimizer, train_loader, model_name: str) -> float:
        """
        Execute light training loop.
        
        Returns:
            Final average loss
        """
        model.train()
        max_batches = self.config.get('max_batches_per_epoch', 50)
        
        # Reduce batch size for memory-intensive models
        if 'siamese' in model_name.lower():
            max_batches = min(max_batches, 25)
        
        final_loss = 0.0
        
        for epoch in range(self.config['num_epochs']):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                
                images = batch['image'].to(self.config['device'])
                masks = batch['mask'].to(self.config['device'])
                
                optimizer.zero_grad()
                
                if self.config['use_amp']:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model(images)
                        loss = loss_fn(outputs, masks)
                else:
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                
                loss.backward()
                
                if self.config.get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['gradient_clip']
                    )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            final_loss = avg_loss
            print(f"  Epoch {epoch+1}/{self.config['num_epochs']}: Loss={avg_loss:.4f}")
        
        return final_loss
    
    def _cleanup_model(self, model, loss_fn, optimizer):
        """Cleanup model resources."""
        del model, loss_fn, optimizer
        self.gpu_mgr.cleanup()
    
    def _print_summary(self, model_names: list):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        passed = [m for m, r in self.results.items() if r['status'] == 'passed']
        failed = [m for m, r in self.results.items() if r['status'] == 'failed']
        
        print(f"\nPassed: {len(passed)}/{len(model_names)}")
        for model_name in passed:
            time_taken = self.results[model_name]['time']
            final_loss = self.results[model_name]['final_loss']
            print(f"  [PASS] {model_name.upper()} ({time_taken:.1f}s, Loss: {final_loss:.4f})")
        
        if failed:
            print(f"\nFailed: {len(failed)}/{len(model_names)}")
            for model_name in failed:
                print(f"  [FAIL] {model_name.upper()}")
                error_msg = self.results[model_name]['error']
                # Truncate long error messages
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"    Error: {error_msg}")
            print("\nWARNING: Fix failed models before proceeding to full training!")
        else:
            print("\nAll models validated successfully! Ready for full training.")
        
        print("="*80)
    
    def get_passed_models(self) -> list:
        """
        Get list of models that passed validation.
        
        Returns:
            List of model names that passed
        """
        return [m for m, r in self.results.items() if r['status'] == 'passed']
    
    def get_failed_models(self) -> list:
        """
        Get list of models that failed validation.
        
        Returns:
            List of model names that failed
        """
        return [m for m, r in self.results.items() if r['status'] == 'failed']
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all validation results.
        
        Returns:
            Dictionary of all validation results
        """
        return self.results
