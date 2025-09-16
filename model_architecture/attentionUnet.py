import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """Attention Gate module for focusing on relevant features"""
    
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): Number of feature maps in gating signal (from decoder)
            F_l (int): Number of feature maps in encoder feature maps 
            F_int (int): Number of intermediate feature maps
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (lower resolution)
            x: Encoder feature maps (higher resolution)
        Returns:
            Attention-weighted encoder features
        """
        # Get attention weights
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample gating signal to match encoder feature map size
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention weights to encoder features
        return x * psi


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with attention gate then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        # Attention gate
        self.attention = AttentionGate(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)

    def forward(self, x1, x2):
        """
        Args:
            x1: Decoder feature maps (to be upsampled)
            x2: Encoder feature maps (skip connection)
        """
        # Upsample decoder features
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Apply attention gate
        x2_att = self.attention(g=x1, x=x2)
        
        # Concatenate attention-weighted encoder features with decoder features
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for burn severity segmentation
    
    Adds attention gates to skip connections to focus on relevant features
    and suppress irrelevant regions in encoder feature maps.
    
    Args:
        n_channels (int): Number of input channels (6 for pre+post fire images)
        n_classes (int): Number of output classes (5 for burn severity levels)
        bilinear (bool): Whether to use bilinear interpolation or transpose convs
    """
    
    def __init__(self, n_channels=6, n_classes=5, bilinear=False):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (Expanding Path) with Attention Gates
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention-gated skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
    
    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'AttentionUNet',
            'input_channels': self.n_channels,
            'output_classes': self.n_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def test_attention_unet():
    """Test function to verify Attention UNet architecture"""
    print("Testing Attention UNet architecture...")
    
    # Create model
    model = AttentionUNet(n_channels=6, n_classes=5)
    
    # Print model info
    info = model.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Input channels: {info['input_channels']}")
    print(f"Output classes: {info['output_classes']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.1f} MB")
    
    # Test with dummy input
    dummy_input = torch.randn(2, 6, 256, 256)  # Batch size 2, 6 channels, 256x256
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    
    print("Attention UNet test completed successfully!")
    return model


if __name__ == "__main__":
    test_attention_unet()