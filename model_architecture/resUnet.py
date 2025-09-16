import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection - adjust dimensions if needed
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out


class ResDoubleConv(nn.Module):
    """Residual Double Convolution block"""
    
    def __init__(self, in_channels, out_channels):
        super(ResDoubleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += residual
        out = self.relu2(out)
        
        return out


class ResDown(nn.Module):
    """Downscaling with maxpool then residual double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class ResUp(nn.Module):
    """Upscaling then residual double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResDoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    """
    Residual U-Net architecture for burn severity segmentation
    
    Incorporates residual connections within the convolution blocks to improve
    gradient flow and enable training of deeper networks.
    
    Args:
        n_channels (int): Number of input channels (6 for pre+post fire images)
        n_classes (int): Number of output classes (5 for burn severity levels)
        bilinear (bool): Whether to use bilinear interpolation or transpose convs
    """
    
    def __init__(self, n_channels=6, n_classes=5, bilinear=False):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Contracting Path) with Residual Blocks
        self.inc = ResDoubleConv(n_channels, 64)
        self.down1 = ResDown(64, 128)
        self.down2 = ResDown(128, 256)
        self.down3 = ResDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = ResDown(512, 1024 // factor)
        
        # Decoder (Expanding Path) with Residual Blocks
        self.up1 = ResUp(1024, 512 // factor, bilinear)
        self.up2 = ResUp(512, 256 // factor, bilinear)
        self.up3 = ResUp(256, 128 // factor, bilinear)
        self.up4 = ResUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder with residual connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with residual connections and skip connections
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
            'model_name': 'ResUNet',
            'input_channels': self.n_channels,
            'output_classes': self.n_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def test_resunet():
    """Test function to verify ResUNet architecture"""
    print("Testing ResUNet architecture...")
    
    # Create model
    model = ResUNet(n_channels=6, n_classes=5)
    
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
    
    print("ResUNet test completed successfully!")
    return model


if __name__ == "__main__":
    test_resunet()