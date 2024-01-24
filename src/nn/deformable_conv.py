import torch
from torch import nn
from torch.nn.modules.utils import _single, _reverse_repeat_tuple
from typing import Union, Literal, Callable
from torch.nn.parameter import Parameter
import math
from torch.nn import functional as F
from src.interp.linear import linear_interpolation


EPS = 1e-9

class GlobalLayerNormalization(nn.Module):
    def __init__(self, channel_size) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        self.gamma.data.fill_(1)
        self.beta.data.zero_()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = (
            (torch.pow(x - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        )
        gln_x = self.gamma * (x - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gln_x

class DeformableConvolution1d(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[int, Literal['valid', 'same']] = 'valid',
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'reflect',
                 device: str = 'cpu',
                 interpolation_func: Callable = linear_interpolation,
                 unconstrained: str = None,
                 *args,
                 **kwargs) -> None:
        
        self.device = device
        self.interpolation_func = interpolation_func
        padding_ = padding if isinstance(padding, str) else _single(padding)
        stride_ = _single(stride)
        dilation_ = _single(dilation)
        kernel_size_ = _single(kernel_size)
        
        super().__init__(*args, **kwargs)
        
        if groups < 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('input channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out channels must be divisible by groups')
        
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError('invalid padding string, you must use valid or same')
            if padding == 'same' and any(s != 1 for s in stride_):
                raise ValueError('padding=same is not supported for strided convolutions')
            
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError('invalid padding mode, you must use zeros, reflect, replicate or circular')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding_
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
            if padding == 'same':
                for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
            
        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        
        self.dilated_positions = torch.linspace(
            0, dilation * kernel_size - dilation, kernel_size
        )
        
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        if not unconstrained == None:
            self.unconstrained = unconstrained
            
        self.reset_parameters()
        self.to(device)
        
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding = {padding}'
        # if self.dilation != (1,) * len(self.dilation):
        s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'

        return s.format(**self.__dict__)
    
    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'
            
    def forward(self,
                x: torch.Tensor,
                offsets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        in_shape = x.shape
        if self.padding_mode != 'zeros':
            x = F.pad(
                x,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode
            )
        elif self.padding == 'same':
            x = F.pad(
                x,
                self._reversed_padding_repeated_twice,
                mode='constant',
                value=0
            )
            
        if not self.device == offsets.device:
            self.device = offsets.device
        if self.dilated_positions.device != self.device:
            self.dilated_positions = self.dilated_positions.to(self.device)
            
        if 'unconstrained' in self.__dict__.keys():
            x = self.interpolation_func(
                x,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device,
                unconstrained=self.unconstrained
            )
        else:
            x = self.interpolation_func(
                x,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device
            )
            
        x = x.flatten(-2, -1)
        output = F.conv1d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=self.kernel_size,
            groups=self.groups
        )
        
        if self.padding == 'same':
            assert in_shape[-1] == output.shape[-1], f'input length {in_shape} and output length {output.shape} do not match'

        return output
        
class LeakySineLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        return torch.where(x < 0, torch.sin(x)**2 + x, 0.1 * (torch.sin(x) ** 2 + x))

class PackedDeformableConvolution1d(DeformableConvolution1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[int, Literal['valid', 'same']] = 'valid',
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 padding_mode: str = 'reflect',
                 offset_groups: int = 1,
                 device: str = 'cpu',
                 interpolation_func: Callable = linear_interpolation,
                 unconstrained: str = None,
                 *args, **kwargs) -> None:
        
        assert offset_groups in [1, in_channels], 'offset groups only implemented for 1 or in_channels'
        
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode,
                         device,
                         interpolation_func,
                         unconstrained,
                         *args, **kwargs)
        
        self.offset_groups = offset_groups

        self.offset_dconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias
        )
        # self.offset_dconv_norm = GlobalLayerNormalization(
        #     in_channels
        # )
        self.offset_dconv_prelu = LeakySineLU()
        
        
        self.offset_pconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=kernel_size*offset_groups,
            kernel_size=1,
            stride=1,
            bias=bias
        )
        # self.offset_pconv_norm = GlobalLayerNormalization(
        #     kernel_size * offset_groups
        # )
        self.offset_pconv_prelu = LeakySineLU()

        self.device = device
        self.to(device)

        torch.nn.init.constant_(self.offset_dconv.weight, 0.)
        torch.nn.init.constant_(self.offset_pconv.weight, 0.)

        if bias:
            torch.nn.init.constant_(self.offset_dconv.bias, 1.)
            torch.nn.init.constant_(self.offset_pconv.bias, 1.)

        self.offset_dconv.register_backward_hook(self._set_lr)
        self.offset_pconv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))    
    
    def forward(self, x: torch.Tensor, with_offsets: bool = False) -> torch.Tensor:
        offsets = self.offset_dconv(x)
        offsets = self.offset_dconv_prelu(offsets)

        self.device = x.device

        assert str(x.device) == str(self.device), 'x and the deformable conv must be on same device'
        # assert str(x.device) == str(offsets.device), 'x and offsets must be on same device'

        offsets = self.offset_pconv(x)
        offsets = self.offset_pconv_prelu(offsets)
        offsets = offsets.unsqueeze(0).chunk(self.offset_groups, dim=2)
        offsets = torch.vstack(offsets).moveaxis((0, 2), (1, 3))

        if with_offsets:
            return super().forward(x, offsets), offsets
        else:
            return super().forward(x, offsets)
