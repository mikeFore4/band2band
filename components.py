import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class ResBlockUp(nn.Module):
    """
    A Residual Upsampling Block including convolutional layers and instance
    normalization. The design is based roughly on the upsampling residual block
    used in https://arxiv.org/pdf/1809.11096v2.pdf. This implementation
    includes suggestions from https://arxiv.org/pdf/1905.08233v1.pdf
    """

    def __init__(self, in_channels, out_channels, scale_factor=2, mode='bilinear',
            internal_embeddings=False, num_classes=None):
        """
        Initialization function for the ResBlockUp module.

        ...

        Inputs
        ---------
        in_channels : int
            number of input channels to block
        out_channels : int
            desired number of output channels from block
        scale_factor : int
            upsampling factor to be used by torch.nn.functional.interpolate
            (default = 2)
        mode : str
            upsampling algorithm to be used by torch.nn.functional.interpolate
            (default = 'bilinear')
        internal_embeddings : bool
            indicates whether modulation parameters for normalization be generated
            from internal embedding layers or inputted externally during forward
            passes (default = False)
        num_classes : int
            only used when internal_embeddings is True. Number of embedding
            layers to create internally
        """

        super(ResBlockUp, self).__init__()

        self.internal_embeddings = internal_embeddings
        self.scale_factor = scale_factor
        self.mode = mode
        self.in1 = AdaIN(in_channels, internal_embeddings, num_classes)
        self.conv_r1 = spectral_norm(nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))

        self.in2 = AdaIN(out_channels, internal_embeddings, num_classes)
        self.conv_r2 = spectral_norm(nn.Conv2d(in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))

        self.conv_l1 = spectral_norm(nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1))

    def forward(self, x, gamma1=None, gamma2=None, beta1=None, beta2=None, class_idx=None):
        """
        Forward pass for ResBlockUp Module

        ...

        Inputs
        ------
        x : torch.tensor
            input features (should have channels equal to self.in_channels)
        gamma1 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the multiplicative modulation parameters for the first
            instance normalization layer (default = None)
        gamma2 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the multiplicative modulation parameters for the second
            instance normalization layer
        beta1 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the additive modulation parameters for the first
            instance normalization layer
        beta2 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the additive modulation parameters for the second
            instance normalization layer
        class_idx : torch.tensor
            required when self.internal_embeddings is True. Contains tensor of
            indexes corresponding to internal embedding layers
        """

        #pass through right side
        if self.internal_embeddings:
            out = self.in1(x, class_idx=class_idx)
        else:
            out = self.in1(x, gammas=gamma1, betas=beta1)
        out = F.relu(out)
        out = F.interpolate(out,scale_factor=self.scale_factor, mode=self.mode)
        out = self.conv_r1(out)
        if self.internal_embeddings:
            out = self.in2(out, class_idx=class_idx)
        else:
            out = self.in2(out, gammas=gamma2, betas=beta2)
        out = F.relu(out)
        out = self.conv_r2(out)

        #pass through left side
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv_l1(x)

        return x+out

class ResBlockDown(nn.Module):
    """
    A Residual Downsampling Block including convolutional layers and instance
    normalization. The design is based roughly on the downsampling residual block
    used in https://arxiv.org/pdf/1809.11096v2.pdf. This implementation
    includes suggestions from https://arxiv.org/pdf/1905.08233v1.pdf
    """

    def __init__(self, in_channels, out_channels, pooling_factor=2, internal_embeddings=False,
            num_classes=None):
        """
        Initialization function for the ResBlockDown module.

        ...

        Inputs
        ---------
        in_channels : int
            number of input channels to block
        out_channels : int
            desired number of output channels from block
        pooling_factor : int
            upsampling factor to be used by torch.nn.functional.avg_pool2d
            (default = 2)
        internal_embeddings : bool
            indicates whether modulation parameters for normalization be generated
            from internal embedding layers or inputted externally during forward
            passes (default = False)
        num_classes : int
            only used when internal_embeddings is True. Number of embedding
            layers to create internally
        """

        super(ResBlockDown, self).__init__()

        self.internal_embeddings = internal_embeddings
        self.pooling_factor = pooling_factor
        self.in1 = AdaIN(in_channels, internal_embeddings, num_classes)
        self.conv_r1 = spectral_norm(nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels,
                kernel_size=3, stride=1, padding=1))
        self.in2 = AdaIN(out_channels, internal_embeddings, num_classes)
        self.conv_r2 = spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1))

        self.conv_l1 = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1))

    def forward(self, x, gamma1=None, gamma2=None, beta1=None, beta2=None, class_idx=None):
        """
        Forward pass for ResBlockDown Module

        ...

        Inputs
        ------
        x : torch.tensor
            input features (should have channels equal to self.in_channels)
        gamma1 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the multiplicative modulation parameters for the first
            instance normalization layer (default = None)
        gamma2 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the multiplicative modulation parameters for the second
            instance normalization layer
        beta1 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the additive modulation parameters for the first
            instance normalization layer
        beta2 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the additive modulation parameters for the second
            instance normalization layer
        class_idx : torch.tensor
            required when self.internal_embeddings is True. Contains tensor of
            indexes corresponding to internal embedding layers
        """

        #right side
        out = F.relu(x)
        if self.internal_embeddings:
            out = self.in1(out, class_idx=class_idx)
        else:
            out = self.in1(out, gammas=gamma1, betas=beta1)
        out = self.conv_r1(out)
        out = F.relu(out)
        if self.internal_embeddings:
            out = self.in2(out, class_idx=class_idx)
        else:
            out = self.in2(out, gammas=gamma2, betas=beta2)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out,self.pooling_factor)

        #left side
        x = self.conv_l1(x)
        x = F.avg_pool2d(x,self.pooling_factor)

        return x+out

class ResBlock(nn.Module):
    """
    A Residual Block including convolutional layers and instance
    normalization. The design is based roughly on the down and up sampling
    residual blocks used in https://arxiv.org/pdf/1809.11096v2.pdf. This
    implementation includes suggestions from https://arxiv.org/pdf/1905.08233v1.pdf
    """

    def __init__(self, in_ channels, out_channels, internal_embeddings=False,
            num_classes=None):
        """
        Initialization function for the ResBlock module.

        ...

        Inputs
        ---------
        in_channels : int
            number of input channels to block
        out_channels : int
            desired number of output channels from block
        internal_embeddings : bool
            indicates whether modulation parameters for normalization be generated
            from internal embedding layers or inputted externally during forward
            passes (default = False)
        num_classes : int
            only used when internal_embeddings is True. Number of embedding
            layers to create internally
        """

        super(ResBlock, self).__init__()

        self.internal_embeddings = internal_embeddings
        self.in1 = AdaIN(in_channels, internal_embeddings, num_classes)
        self.conv_r1 = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1))
        self.in2 = AdaIN(out_channels, internal_embeddings, num_classes)
        self.conv_r2 = spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1))

        self.conv_l1 = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1))

    def forward(self, x, gammas=None, betas=None, class_idx=None):
        """
        Forward pass for ResBlock Module

        ...

        Inputs
        ------
        x : torch.tensor
            input features (should have channels equal to self.in_channels)
        gamma1 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the multiplicative modulation parameters for the first
            instance normalization layer (default = None)
        gamma2 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the multiplicative modulation parameters for the second
            instance normalization layer
        beta1 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the additive modulation parameters for the first
            instance normalization layer
        beta2 : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the additive modulation parameters for the second
            instance normalization layer
        class_idx : torch.tensor
            required when self.internal_embeddings is True. Contains tensor of
            indexes corresponding to internal embedding layers
        """

        if not self.internal_embeddings:
            gamma_shape = [dim for dim in gammas.shape]
            assert gamma_shape == [2,x.shape[0],x.shape[1]]
            beta_shape = [dim for dim in betas.shape]
            assert beta_shape == [2,x.shape[0],x.shape[1]]

        #right side
        out = F.relu(x)
        if self.internal_embeddings:
            out = self.in1(out, class_idx=class_idx)
        else:
            out = self.in1(out, gammas=gammas[0,:,:], betas=betas[0,:,:])
        out = self.conv_r1(out)
        out = F.relu(out)
        if self.internal_embeddings:
            out = self.in2(out, class_idx=class_idx)
        else:
            out = self.in2(out, gammas=gammas[1,:,:], betas=betas[1,:,:])
        out = self.conv_r2(out)

        #left side
        x = self.conv_l1(x)

        return x+out

class AdaIN(nn.Module):
    """
    Implementation of Adaptive Instance Normalization for vision tasks
    """

    def __init__(self, channels, internal_embeddings=False, num_classes=None):
        """
        Initialization method for AdaIN module

        ...

        Inputs
        ------
        channels : int
            number of channels for this layer, both input and output
        internal_embeddings : bool
            indicates whether to build in embedding
            layer to compute gammas and betas for denormalizatin transform inside
            this module (True), or to take gammas and betas as input in a forward
            pass (False) (default = False)
        num_classes: int
            number of embeddings to generate if internal_embeddings
            is True (default=None)
        """

        super(AdaIN, self).__init__()
        self.internal_embeddings = internal_embeddings
        if self.internal_embeddings:
            self.gamma_emb = nn.Embedding(num_embeddings=num_classes,
                    embedding_dim=channels)
            self.beta_emb = nn.Embedding(num_embeddings=num_classes,
                    embedding_dim=channels)
        else:
            self.gamma_emb = None
            self.beta_emb = None
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)

    def forward(self, x, gammas=None, betas=None, class_idx=None):
        """
        Forward pass for AdaIN Module

        ...

        Inputs
        ------
        x : torch.tensor
            input features (should have channels equal to self.in_channels)
        gammas : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the multiplicative modulation parameters
            (default = None)
        betas : torch.tensor
            required when self.internal_embeddings is False. Gives a tensor
            providing the additive modulation parameters
            (default = None)
        class_idx : torch.tensor
            required when self.internal_embeddings is True. Contains tensor of
            indexes corresponding to internal embedding layers
        """
        x = self.instance_norm(x)
        if self.internal_embeddings:
            assert len(class_idx.shape) == 2
            assert class_idx.shape[0] == x.shape[0]
            assert class_idx.shape[1] == 1
            gammas = self.gamma_emb(class_idx)
            betas = self.beta_emb(class_idx)

        gammas = gammas.view(x.shape[0],x.shape[1],1,1)
        betas = betas.view(x.shape[0],x.shape[1],1,1)
        x = x*gammas + betas

        return x

class SpectralNormConv2d(nn.Module):
    """
    Simple module that combines Conv2d with spectral normalization. Can be used
    in place of Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, bias=True):
        """
        Initialization for SpectralNormConv2d layer

        ...

        Inputs
        ------
        in_channels : int
            number of input channels for input tensor
        out_channels : int
            number of channels for output tensor
        kernel_size : int
            spatial dimension of square convolutional filter
        stride : int
            step distance for convolutional filter (default = 1)
        padding : int
            padding to apply to input tensor for convolution (default = 0)
        bias : bool
            indicates whether to add bias to output (default = True)
        """

        super(SpectralNormConv2d, self).__init__()

        self.SNConv = spectral_norm(nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias))

    def forward(self, x):
        """
        Forward pass for SpectralNormConv2d module

        ...

        Inputs
        ------
        x : torch.tensor
        """

        return self.SNConv(x)

class SelfAttention(nn.Module):
    """
    This class code is copied directly from this repo:
    https://github.com/ajbrock/BigGAN-PyTorch and is based on the blocks from
    the original SA-GAN paper: https://arxiv.org/pdf/1805.08318v2.pdf
    Implements self attention module
    """
    def __init__(self, ch, which_conv=SpectralNormConv2d, name='attention'):
        """
        Initialization for SelfAttention module

        ...

        Inputs
        ------
        ch : int
            channel multiplier
        which_conv : torch.nn.Module
            what type of convolutional layer to use
            (default = SpectralNormConv2d)
        name : str
            (default = 'attention')
        """

        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        """
        Forward pass for SelfAttention module

        ...

        Inputs
        ------
        x : torch.tensor
        """
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))

        return self.gamma * o + x

class SequentialConditional(nn.Sequential):
    """
    Subclass of nn.Sequential that allows a specific additional input of
    "class_idx" to be passed to the modules in the sequential layer
    """

    def forward(self, x, class_idx):
        for module in self._modules.values():
            x = module(x, class_idx=class_idx)

        return x
