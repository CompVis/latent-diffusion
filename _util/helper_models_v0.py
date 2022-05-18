

from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)
class AbsoluteValue(nn.Module):
    def __init__(self):
        super(AbsoluteValue, self).__init__()
        return
    def forward(self, x):
        return torch.abs(x)
class MaskedImageLossL1(nn.Module):
    def __init__(self):
        super(MaskedImageLossL1, self).__init__()
        return
    def forward(self, alpha, gt, pred):
        return (alpha * (gt-pred).abs().mean(dim=1, keepdim=True)).mean()

class PositionalEncoding(nn.Module):
    def __init__(self, phases, freq_min=1, freq_max=10000):
        super(PositionalEncoding, self).__init__()
        self.phases = phases
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.size = 2*phases  # for sin and cos
        ls = np.linspace(0, 1, phases)
        mult = torch.tensor(np.exp(
            ls * np.log(freq_max) + (1-ls) * np.log(freq_min)
        )) * (2*np.pi)
        self.register_buffer('multiplier', mult)
        return
    def forward(self, x):
        t = x.unsqueeze(-1) * self.multiplier
        t = torch.stack([
            torch.sin(t),
            torch.cos(t),
        ], dim=-1).flatten(-2)
        return t.float()

class ResBlock(nn.Module):
    def __init__(self,
            depth, channels, kernel,
            channels_in=None,  # in case different from channels
            activation=nn.ReLU,
            normalization=nn.BatchNorm2d,
                ):
        # activation()
        # normalization(channels)
        super(ResBlock, self).__init__()
        self.depth = depth
        self.channels = channels
        self.channels_in = channels_in
        self.kernel = kernel
        self.activation = activation
        self.normalization = normalization

        # create sequential network
        od = OrderedDict()
        for i in range(depth):
            chin = channels_in \
                if channels_in is not None and i==0 \
                else channels
            od[f'conv{i}'] = nn.Conv2d(
                chin, channels,
                kernel_size=kernel, padding=kernel//2,
                bias=True, padding_mode='replicate',
            )
            if activation is not None:
                od[f'act{i}'] = activation()
            if normalization is not None:
                od[f'norm{i}'] = normalization(channels)
        self.net = nn.Sequential(od)

        # last activation/normalization
        od_tail = OrderedDict()
        if activation is not None:
            od_tail[f'act{depth}'] = activation()
        if normalization is not None:
            od_tail[f'norm{depth}'] = normalization(channels)
        self.net_tail = nn.Sequential(od_tail)
        return

    def forward(self, x):
        if self.channels_in is None:
            return self.net_tail(x + self.net(x))
        else:
            head = self.net[0](x)
            t = head
            for body in self.net[1:]:
                t = body(t)
            return self.net_tail(head + t)


class Interpolator2d(nn.Module):
    def __init__(self, size=None, mode='nearest'):
        # will work as long as batch dim matches
        # modes: nearest, bilinear, bicubic, area (downscaling)
        super(Interpolator2d, self).__init__()
        self.size = (size, size) if type(size)==int else size
        self.mode = mode
        return

    def forward(self, x, size=None, mode=None):
        # local vars override defaults
        size = size or self.size
        size = (size, size) if type(size)==int else size
        mode = mode or self.mode
        return torch.cat([
            TF.interpolate(
                t, size=size, mode=mode,
                # align_corners=not mode in ['nearest', 'area'],
            )
            if t.shape[-2:]!=self.size else t
            for t in x if t is not None
        ], dim=1)


class Injector(nn.Module):
    def __init__(self,
            size, # input image height/width
            depth, # of the resnet
            channels_input,
            channels_preprocess, # on input
            channels_input_aux, # list
            channels_resnet,
            kernel,
            activation=nn.ReLU,
            normalization=nn.BatchNorm2d,
            interpolation_mode='bicubic',
                ):
        # activation()
        # normalization(channels)
        super(Injector, self).__init__()
        self.size = (size, size) if type(size)==int else size
        self.depth = depth
        self.channels_input = channels_input
        self.channels_preprocess = channels_preprocess
        self.channels_input_aux = channels_input_aux
        self.channels_resnet = channels_resnet
        self.kernel = kernel
        self.activation = activation
        self.normalization = normalization
        self.interpolation_mode = interpolation_mode

        # adds activation and normalization to tail of conv
        def _conv_and_tail(c_in, c_out):
            od = OrderedDict()
            od['conv'] = nn.Conv2d(
                c_in, c_out,
                kernel_size=kernel, padding=kernel//2,
                padding_mode='replicate',
            )
            if activation is not None:
                od['act'] = activation()
            if normalization is not None:
                od['norm'] = normalization(c_out)
            return nn.Sequential(od)

        # create input preprocessor
        self.net_preprocessor = _conv_and_tail(
            channels_input, channels_preprocess,
        )

        # create aux input interpolator
        self.net_interpolator = Interpolator2d(
            size, mode=interpolation_mode,
        )

        # create converter to resnet channels
        self.net_resnet_converter = _conv_and_tail(
            sum(channels_input_aux) + channels_preprocess,
            channels_resnet,
        )

        # create resnet
        self.net_resnet = ResBlock(
            depth, channels_resnet, kernel,
            activation=activation, normalization=normalization,
        )
        return

    def forward(self, x, x_aux):
        # preprocess input and cat with aux input
        t = self.net_interpolator([x,])
        t = self.net_preprocessor(t)
        t = self.net_interpolator(x_aux + [t,])

        # feed through resnet
        t = self.net_resnet_converter(t)
        t = self.net_resnet(t)
        return t
