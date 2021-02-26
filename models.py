import torch.nn as nn
from components import ResBlockUp, ResBlockDown, ResBlock, SelfAttention, SequentialConditional

class Encoder(nn.Module):

    def __init__(self, input_channels, down_blocks, const_blocks, num_classes, pooling_factor=2,
            internal_embeddings=True, first_out_channels=32,
            down_channel_multiplier=2, const_channel_multiplier=1):
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
                        internal_embeddings=internal_embeddings,
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
                        internal_embeddings=internal_embeddings,
                        num_classes=num_classes
                        )
                    )

        self.model=SequentialConditional(*model)

    def forward(self, x, class_idx):
         return self.model(x, class_idx=class_idx)

class Decoder(nn.Module):

    def __init__(self, input_channels, up_blocks, const_blocks, num_classes,
            scale_factor=2, mode='bilinear',
            internal_embeddings=True, up_channel_divisor=2,
            const_channel_divisor=1, output_channels=3):
        super(Decoder, self).__init__()

        model = []
        in_channels = input_channels
        for i in range(const_blocks):
            out_channels = int(in_channels/const_channel_divisor)
            model.append(
                    ResBlock(
                        in_channels,
                        out_channels,
                        internal_embeddings=internal_embeddings,
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
                        internal_embeddings=internal_embeddings,
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
                internal_embeddings=internal_embeddings,
                num_classes=num_classes
                )
            )

        self.model = SequentialConditional(*model)

    def forward(self, x, class_idx):
         return self.model(x, class_idx=class_idx)
