import torch.nn as nn
import torch.nn.functional as F
from components import ResBlockUp, ResBlockDown, ResBlock, SelfAttention, SequentialConditional

class Encoder(nn.Module):
    """
    Encoder module for band2band translation
    """

    def __init__(self, input_channels, down_blocks, const_blocks, num_classes, pooling_factor=2, first_out_channels=32,
            down_channel_multiplier=2, const_channel_multiplier=1):
        """
        Initizalizer for Encoder

        ...

        Inputs
        ------
            input_channels : int
                number of channels for input images
            down_blocks : int
                number of downsampling residual blocks
            const_blocks : int
                number of residual blocks that preserve the spatial dimension
            num_classes : int
                number of separate internal embeddings to generate
            pooling_factor : int
                upsampling factor to be used by torch.nn.functional.avg_pool2d
                (default = 2)
            first_out_channels : int
                number of output channels for first convolutional layer
                (default = 32)
            down_channel_multiplier : int
                rate used to increase number of channels with each downsampling
                block (default = 2)
            const_channel_multiplier : int
                rate used to increase the number of channels with each const
                block (default = 1)
        """

        super(Encoder, self).__init__()

        model = []
        out_channels = first_out_channels
        in_channels = input_channels
        for i in range(down_blocks):
            if i != 0:
                in_channels = out_channels
                out_channels *= down_channel_multiplier
            model.append(
                    ResBlockDown(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        pooling_factor=pooling_factor,
                        internal_embeddings=True,
                        num_classes=num_classes
                        )
                    )

        for i in range(const_blocks):
            in_channels = out_channels
            out_channels *= const_channel_multiplier
            model.append(
                    ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        internal_embeddings=True,
                        num_classes=num_classes
                        )
                    )

        self.model=SequentialConditional(*model)

    def forward(self, x, class_idx):
        """
        Forward pass for Encoder module

        ...

        Inputs
        ------
        x : torch.tensor
            input image
        class_idx : torch.tensor
            band number of input image
        """

        return self.model(x, class_idx=class_idx)

class Decoder(nn.Module):
    """
    Decoder module for band2band translation
    """

    def __init__(self, input_channels, up_blocks, const_blocks, num_classes,
            output_channels, scale_factor=2, mode='bilinear', up_channel_divisor=2,
            const_channel_divisor=1):
        """
        Initizalizer for Decoder

        ...

        Inputs
        ------
            input_channels : int
                number of channels for input images
            up_blocks : int
                number of upsampling residual blocks
            const_blocks : int
                number of residual blocks that preserve the spatial dimension
            num_classes : int
                number of separate internal embeddings to generate
            output_channels : int
                number of channels to output in final generation
            scale_factor : int
                upsampling factor to be used by torch.nn.functional.interpolate
                (default = 2)
            mode : str
                upsampling algorithm to be used by torch.nn.functional.interpolate
                (default = 'bilinear')
            up_channel_divisor : int
                rate used to decrease number of channels with each upsampling
                block (default = 2)
            const_channel_divisor : int
                rate used to decrease the number of channels with each const
                block (default = 1)
        """

        super(Decoder, self).__init__()

        model = []
        in_channels = input_channels
        for i in range(const_blocks):
            out_channels = int(in_channels/const_channel_divisor)
            model.append(
                    ResBlock(
                        in_channels,
                        out_channels,
                        internal_embeddings=True,
                        num_classes=num_classes
                        )
                    )
            in_channels=out_channels

        for i in range(up_blocks-1):
            out_channels = int(in_channels/up_channel_divisor)
            model.append(
                    ResBlockUp(
                        in_channels,
                        out_channels,
                        scale_factor=scale_factor,
                        mode=mode,
                        internal_embeddings=True,
                        num_classes=num_classes
                        )
                    )
            in_channels=out_channels

        model.append(
            ResBlockUp(
                in_channels,
                output_channels,
                scale_factor=scale_factor,
                mode=mode,
                internal_embeddings=True,
                num_classes=num_classes
                )
            )

        self.model = SequentialConditional(*model)

    def forward(self, x, class_idx):
        """
        Forward pass for Decoder module
        """

        return self.model(x, class_idx=class_idx)
