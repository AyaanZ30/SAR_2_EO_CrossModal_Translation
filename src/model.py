import torch
import torch.nn as nn

CONV_KERNEL_SIZE = 4
CONV_STRIDE_DOWNSAMPLE = 2
CONV_STRIDE_SAME = 1
CONV_PADDING = 1

LEAKY_RELU_SLOPE = 0.2
DECODER_DROPOUT_PROB = 0.5
NUM_DROPOUT_DECODER_LAYERS = 3

class UNetDownBlock(nn.Module):
    """
    One U-Net encoder step: halves spatial resolution, usually doubles channels.
    Conv(4x4, stride 2) -> [InstanceNorm] -> LeakyReLU(0.2)

    `normalize=False` is used for the very first encoder block: normalizing the
    raw input SAR intensities before the network has done anything to them is
    unnecessary and, per the original pix2pix design, is skipped.
    """
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels,
            kernel_size = CONV_KERNEL_SIZE, stride = CONV_STRIDE_DOWNSAMPLE,
            padding = CONV_PADDING, bias = not normalize),
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(LEAKY_RELU_SLOPE))
        self.block = nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size = CONV_KERNEL_SIZE, stride = CONV_STRIDE_DOWNSAMPLE,
            padding = CONV_PADDING, bias = False),

            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(DECODER_DROPOUT_PROB))
        self.block = nn.Sequential(*layers)

    def forward(self, x : torch.Tensor, skip_connections : torch.Tensor) -> torch.Tensor:
        upsampled = self.block(x)

        # channel-wise concatenation
        return torch.cat((upsampled, skip_connections), dim = 1)   # for that scale (for getting direct access to feature maps of finer spatial details)

class UNetGenerator(nn.Module):
    """
    8-down / 8-up U-Net generator. See module docstring for I/O shapes.

    in_channels=1 matches the SAR I/O contract (single-channel VV input).
    out_channels=3 matches RGB EO output.
    base_channels controls model capacity (64 = standard pix2pix default;
    lower it if you need to fit a smaller GPU/VRAM budget).
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels : int = 64):
        super().__init__()
        c = base_channels

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 (1 x 512 x 512)
        self.down_1 = UNetDownBlock(in_channels, c, normalize = False)
        self.down_2 = UNetDownBlock(c, c * 2)
        self.down_3 = UNetDownBlock(c * 2, c * 4)
        self.down_4 = UNetDownBlock(c * 4, c * 8)
        self.down_5 = UNetDownBlock(c * 8, c * 8)
        self.down_6 = UNetDownBlock(c * 8, c * 8)
        self.down_7 = UNetDownBlock(c * 8, c * 8)
        self.down_8 = UNetDownBlock(c * 8, c * 8, normalize = False)

        # mirrors the encoder
        self.up_1 = UNetUpBlock(c * 8, c * 8, use_dropout = True)
        self.up_2 = UNetUpBlock(c * 16, c * 8, use_dropout = True)
        self.up_3 = UNetUpBlock(c * 16, c * 8, use_dropout = True)
        self.up_4 = UNetUpBlock(c * 16, c * 8)
        self.up_5 = UNetUpBlock(c * 16, c * 4)
        self.up_6 = UNetUpBlock(c * 8, c * 2)
        self.up_7 = UNetUpBlock(c * 4, c)

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = c * 2, out_channels = out_channels,      # 128 -> 256 x 3 x 3
                kernel_size = CONV_KERNEL_SIZE, stride = CONV_STRIDE_DOWNSAMPLE,
                padding = CONV_PADDING
            ),
            nn.Tanh()   # output in [-1, 1], matching how EO targets are normalized
        )

    def forward(self, sar_image : torch.Tensor) -> torch.Tensor:
        d1 = self.down_1(sar_image)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        d5 = self.down_5(d4)
        d6 = self.down_6(d5)
        d7 = self.down_7(d6)
        d8 = self.down_8(d7)

        upsample_block_input = downsample_last_block_output = d8

        u1 = self.up_1(upsample_block_input, d7)
        u2 = self.up_2(u1, d6)
        u3 = self.up_3(u2, d5)
        u4 = self.up_4(u3, d4)
        u5 = self.up_5(u4, d3)
        u6 = self.up_6(u5, d2)
        u7 = self.up_7(u6, d1)

        return self.output_layer(u7)

class PatchGANDiscriminator(nn.Module):
    """
    70x70-receptive-field PatchGAN discriminator, CONDITIONAL on the SAR input.

    Judging many local patches independently (a scalar score for each) => forces locally
    realistic texture everywhere    
    
    WHY CONDITIONAL (concatenating the SAR input, not just judging the EO
    image alone): without conditioning, the discriminator can only ask "does
    this look like a plausible EO image in general"
    """
    def __init__(self, sar_channels : int = 1, eo_channels : int = 3, base_channels : int = 64):
        super().__init__()
        c = base_channels
        
        def conv_block(in_ch: int, out_ch: int, stride: int, normalize: bool = True):
            layers = [
                nn.Conv2d(
                    in_channels = in_ch, 
                    out_channels = out_ch, 
                    kernel_size = CONV_KERNEL_SIZE, 
                    stride = stride, 
                    padding = CONV_PADDING, 
                    bias = not normalize
                )
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace = True))
            return layers

        self.model = nn.Sequential(
            # input channels = SAR (condition) + EO (real or generated), concatenated
            *conv_block(in_ch = (sar_channels + eo_channels), out_ch = c, stride = CONV_STRIDE_DOWNSAMPLE, normalize = False),
            *conv_block(c, c * 2, stride = CONV_STRIDE_DOWNSAMPLE),
            *conv_block(c * 2, c * 4, stride = CONV_STRIDE_DOWNSAMPLE),
            *conv_block(c * 4, c * 8, stride = CONV_STRIDE_SAME),
            nn.Conv2d(in_channels = c * 8, out_channels = 1, kernel_size = CONV_KERNEL_SIZE, stride = CONV_STRIDE_SAME, padding = CONV_PADDING)
        )
        
    def forward(self, sar_image : torch.Tensor, eo_image : torch.Tensor) -> torch.Tensor:
        concatenated_input = torch.cat((sar_image, eo_image), dim = 1)
        return self.model(concatenated_input)

if __name__ == "__main__":
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()
    
    dummy_sar = torch.randn(2, 1, 256, 256)
    dummy_eo = torch.randn(2, 3, 256, 256)
    
    generated_eo = generator(dummy_sar)
    patch_predictions = discriminator(dummy_sar, generated_eo)
 
    print(f"generator:     input {tuple(dummy_sar.shape)} -> output {tuple(generated_eo.shape)}")
    print(f"discriminator: output {tuple(patch_predictions.shape)}")
 
    assert generated_eo.shape == (2, 3, 256, 256), "generator output shape is wrong"
    assert patch_predictions.shape[-2:] == (30, 30), (
        f"discriminator should output a ~30x30 patch map (70x70 receptive field), "
        f"got {tuple(patch_predictions.shape[-2:])} instead -- check conv strides"
    )
    print("shape checks passed")