"""
Model architectures for flood detection segmentation
Includes U-Net++, DeepLabV3+, and SegFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import segmentation_models_pytorch as smp


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) architecture
    Paper: https://arxiv.org/abs/1807.10165
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 7,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: Optional[str] = None,
        deep_supervision: bool = False
    ):
        """
        Initialize U-Net++
        
        Args:
            in_channels: Number of input channels (6 for pre+post)
            num_classes: Number of output segmentation classes
            encoder_name: Encoder backbone (resnet34, efficientnet-b0, etc.)
            encoder_weights: Pretrained weights (imagenet, None)
            decoder_channels: Number of channels in decoder layers
            decoder_attention_type: Attention mechanism (None, 'scse')
            deep_supervision: Enable deep supervision
        """
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Use segmentation_models_pytorch implementation
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type
        )
        
        # Modify first conv layer if pretrained for 3 channels
        if encoder_weights is not None and in_channels != 3:
            self._adapt_first_conv(in_channels)
        
        print(f"UNet++ initialized:")
        print(f"  Encoder: {encoder_name}")
        print(f"  Input channels: {in_channels}")
        print(f"  Output classes: {num_classes}")
        print(f"  Pretrained: {encoder_weights}")
        print(f"  Deep supervision: {deep_supervision}")
    
    def _adapt_first_conv(self, in_channels: int):
        """Adapt first convolutional layer for different input channels"""
        # Get first conv layer
        if hasattr(self.model.encoder, 'conv1'):
            old_conv = self.model.encoder.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            # Initialize new conv with repeated weights
            with torch.no_grad():
                # Average weights across input channels
                weight = old_conv.weight.repeat(1, in_channels // 3 + 1, 1, 1)[:, :in_channels, :, :]
                new_conv.weight = nn.Parameter(weight / (in_channels / 3))
                if old_conv.bias is not None:
                    new_conv.bias = nn.Parameter(old_conv.bias.clone())
            
            self.model.encoder.conv1 = new_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture
    Paper: https://arxiv.org/abs/1802.02611
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 7,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36)
    ):
        """
        Initialize DeepLabV3+
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
            encoder_output_stride: Output stride (8 or 16)
            decoder_channels: Number of decoder channels
            decoder_atrous_rates: Atrous rates for ASPP
        """
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates
        )
        
        # Modify first conv layer if needed
        if encoder_weights is not None and in_channels != 3:
            self._adapt_first_conv(in_channels)
        
        print(f"DeepLabV3+ initialized:")
        print(f"  Encoder: {encoder_name}")
        print(f"  Input channels: {in_channels}")
        print(f"  Output classes: {num_classes}")
        print(f"  Output stride: {encoder_output_stride}")
        print(f"  Pretrained: {encoder_weights}")
    
    def _adapt_first_conv(self, in_channels: int):
        """Adapt first convolutional layer"""
        if hasattr(self.model.encoder, 'conv1'):
            old_conv = self.model.encoder.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            with torch.no_grad():
                weight = old_conv.weight.repeat(1, in_channels // 3 + 1, 1, 1)[:, :in_channels, :, :]
                new_conv.weight = nn.Parameter(weight / (in_channels / 3))
                if old_conv.bias is not None:
                    new_conv.bias = nn.Parameter(old_conv.bias.clone())
            
            self.model.encoder.conv1 = new_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)


class SegFormer(nn.Module):
    """
    SegFormer architecture (using pretrained model from transformers)
    Paper: https://arxiv.org/abs/2105.15203
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 7,
        model_name: str = 'nvidia/segformer-b0-finetuned-ade-512-512',
        pretrained: bool = True
    ):
        """
        Initialize SegFormer
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            model_name: Pretrained model name from Hugging Face
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        try:
            from transformers import SegformerForSemanticSegmentation
            
            if pretrained:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            else:
                from transformers import SegformerConfig
                config = SegformerConfig.from_pretrained(model_name)
                config.num_labels = num_classes
                self.model = SegformerForSemanticSegmentation(config)
            
            # Adapt first layer for 6 channels
            if in_channels != 3:
                self._adapt_patch_embed(in_channels)
            
            print(f"SegFormer initialized:")
            print(f"  Model: {model_name}")
            print(f"  Input channels: {in_channels}")
            print(f"  Output classes: {num_classes}")
            print(f"  Pretrained: {pretrained}")
            
        except ImportError:
            print("ERROR: transformers library not installed")
            print("Install with: pip install transformers")
            raise
    
    def _adapt_patch_embed(self, in_channels: int):
        """Adapt patch embedding for different input channels"""
        # SegFormer uses patch embeddings in the encoder
        for i, block in enumerate(self.model.segformer.encoder.patch_embeddings):
            old_proj = block.proj
            new_proj = nn.Conv2d(
                in_channels,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding
            )
            
            with torch.no_grad():
                # Repeat and average weights
                weight = old_proj.weight.repeat(1, in_channels // 3 + 1, 1, 1)[:, :in_channels, :, :]
                new_proj.weight = nn.Parameter(weight / (in_channels / 3))
                new_proj.bias = nn.Parameter(old_proj.bias.clone())
            
            block.proj = new_proj
            
            # Only modify first layer
            break
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        
        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return logits


def create_model(
    model_name: str,
    in_channels: int = 6,
    num_classes: int = 7,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: 'unet++', 'deeplabv3+', or 'segformer'
        in_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name in ['unet++', 'unetplusplus', 'unet_plusplus']:
        return UNetPlusPlus(in_channels=in_channels, num_classes=num_classes, **kwargs)
    
    elif model_name in ['deeplabv3+', 'deeplabv3plus', 'deeplab']:
        return DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes, **kwargs)
    
    elif model_name in ['segformer', 'seg_former']:
        return SegFormer(in_channels=in_channels, num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: unet++, deeplabv3+, segformer")


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")
    
    batch_size = 2
    in_channels = 6
    height = width = 512
    num_classes = 7
    
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Test U-Net++
    print("="*60)
    model = create_model('unet++', encoder_name='resnet34', encoder_weights=None)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test DeepLabV3+
    print("\n" + "="*60)
    model = create_model('deeplabv3+', encoder_name='resnet50', encoder_weights=None)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test SegFormer (skip if transformers not installed)
    print("\n" + "="*60)
    try:
        model = create_model('segformer', pretrained=False)
        y = model(x)
        print(f"Output shape: {y.shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except ImportError:
        print("SegFormer skipped (transformers not installed)")
