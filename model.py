import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=5, base_filters=48, leaky_relu_slope=0.1):
        """
        U-Net architecture for image denoising.

        Args:
            in_channels (int): Number of input channels, default 3
            out_channels (int): Number of output channels, default 3
            depth (int): Depth of the network (number of downsampling layers), default 5
            base_filters (int): Number of filters in the first layer, default 48
            leaky_relu_slope (float): Negative slope for LeakyReLU activation, default 0.1
        """
        super(UNet, self).__init__()
        self.depth = depth
        self.base_filters = base_filters

        self.input_conv = nn.Sequential(
            ConvLeakyReLU(in_channels, base_filters, kernel_size=3, slope=leaky_relu_slope),
            ConvLeakyReLU(base_filters, base_filters, kernel_size=3, slope=leaky_relu_slope),
        )

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            self.encoder_blocks.append(
                ConvLeakyReLU(base_filters, base_filters, kernel_size=3, slope=leaky_relu_slope)
            )

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth):
            if i != depth - 1:
                input_channels = base_filters * 2 if i == 0 else base_filters * 3
                self.decoder_blocks.append(
                    UpsampleBlock(input_channels, base_filters * 2, slope=leaky_relu_slope)
                )
            else:
                input_channels = base_filters * 2 + in_channels
                self.decoder_blocks.append(
                    UpsampleBlock(input_channels, base_filters * 2, slope=leaky_relu_slope)
                )

        # Final output layers
        self.output_conv = nn.Sequential(
            ConvLeakyReLU(2 * base_filters, 2 * base_filters, kernel_size=1, slope=leaky_relu_slope),
            ConvLeakyReLU(2 * base_filters, 2 * base_filters, kernel_size=1, slope=leaky_relu_slope),
            conv1x1(2 * base_filters, out_channels, bias=True),
        )

    def forward(self, x):
        # Store skip connections for U-Net architecture
        skip_connections = []
        skip_connections.append(x)  # Store original input

        features = self.input_conv(x)

        # Encoder path: downsample and store skip connections
        for level, encoder_block in enumerate(self.encoder_blocks):
            features = F.max_pool2d(features, kernel_size=2)  # Downsample by factor of 2

            # Store intermediate features for skip connections (except the deepest level)
            if level != len(self.encoder_blocks) - 1:
                skip_connections.append(features)

            features = encoder_block(features)

        # Decoder path: upsample and combine with skip connections
        for level, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = skip_connections[-(level + 1)]  # Get corresponding skip connection
            features = decoder_block(features, skip_connection)

        output = self.output_conv(features)
        return output


class ConvLeakyReLU(nn.Module):
    """Convolution followed by LeakyReLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, slope=0.1):
        super(ConvLeakyReLU, self).__init__()
        padding = kernel_size // 2  # Maintain spatial dimensions

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class UpsampleBlock(nn.Module):
    """Upsampling block that combines upsampled features with skip connections."""

    def __init__(self, in_channels, out_channels, slope=0.1):
        super(UpsampleBlock, self).__init__()
        self.conv1 = ConvLeakyReLU(in_channels, out_channels, slope=slope)
        self.conv2 = ConvLeakyReLU(out_channels, out_channels, slope=slope)

    def upsample_features(self, x):
        """Upsample feature maps by factor of 2 using nearest neighbor interpolation."""
        batch_size, channels, height, width = x.shape

        # Reshape to insert new dimensions for upsampling
        x = x.reshape(batch_size, channels, height, 1, width, 1)

        # Repeat along height and width dimensions to double the size
        x = x.repeat(1, 1, 1, 2, 1, 2)

        # Reshape back to standard format with doubled spatial dimensions
        upsampled = x.reshape(batch_size, channels, height * 2, width * 2)
        return upsampled

    def forward(self, features, skip_connection):
        upsampled_features = self.upsample_features(features)
        combined_features = torch.cat([upsampled_features, skip_connection], dim=1)

        # Apply convolutions
        output = self.conv1(combined_features)
        output = self.conv2(output)

        return output


def conv1x1(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=bias)
    return layer
