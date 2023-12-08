import torch
import torch.nn as nn


class DeformConv1D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, lr_ratio=1.0):
        """
        Initialize the DeformConv1D module.

        Parameters:
            inc (int): Number of input channels.
            outc (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding added to both sides of the input.
            stride (int): Stride of the convolution.
            bias (bool): If True, adds a learnable bias to the output.
            lr_ratio (float): Learning rate ratio used for gradient scaling.
        """
        super(DeformConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad1d(padding)

        # Offset convolution layer with 1D convolution
        self.offset_conv = nn.Conv1d(inc, kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0)
        self.offset_conv.register_backward_hook(self._set_lr)

        # Regular 1D convolution layer
        self.conv = nn.Conv1d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.lr_ratio = lr_ratio

    def _set_lr(self, module, grad_input, grad_output):
        """
        Backward hook to scale the gradient with the learning rate ratio.

        Parameters:
            module (nn.Module): Current module.
            grad_input (tuple of Tensor): Input gradients.
            grad_output (tuple of Tensor): Output gradients.

        Returns:
            tuple of Tensor: Scaled input gradients.
        """
        new_grad_input = []

        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                new_grad_input.append(grad_input[i] * self.lr_ratio)
            else:
                new_grad_input.append(grad_input[i])

        new_grad_input = tuple(new_grad_input)
        return new_grad_input

    def forward(self, x):
        """
        Forward pass of the deformable convolution.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after deformable convolution.
        """
        offset = self.offset_conv(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1)

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)

        # (b, h, 2N)
        p = p.contiguous().permute(0, 2, 1)

        # Calculate indices for interpolation
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.clamp(q_lt[..., :N], 0, x.size(2) - 1).long()
        q_rb = torch.clamp(q_rb[..., :N], 0, x.size(2) - 1).long()

        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding)],
                         dim=-1).type_as(p)
        mask = mask.detach()

        floor_p = torch.floor(p)
        p = p * (1 - mask) + floor_p * mask
        p = torch.clamp(p[..., :N], 0, x.size(2) - 1)

        g_lt = (1 + (q_lt.type_as(p) - p[..., :N])) * (1 - (q_rb.type_as(p) - p[..., :N]))
        g_rb = (1 - (q_rb.type_as(p) - p[..., :N])) * (1 + (q_lt.type_as(p) - p[..., :N]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb

        x_offset = self._reshape_x_offset(x_offset, ks)

        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        """
        Calculate the indices along the x-axis.

        Parameters:
            N (int): Number of channels.
            dtype (torch.dtype): Data type of the tensor.

        Returns:
            Tensor: Tensor representing indices along the x-axis.
        """
        # Create tensor representing indices along the x-axis
        p_n_x = torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)
        p_n = torch.flatten(p_n_x)
        p_n = p_n.view(1, 2 * N, 1).type(dtype)
        p_n.requires_grad = False

        return p_n

    def _get_p_0(self, h, N, dtype):
        """
        Calculate the indices along the y-axis.

        Parameters:
            h (int): Height of the feature map.
            N (int): Number of channels.
            dtype (torch.dtype): Data type of the tensor.

        Returns:
            Tensor: Tensor representing indices along the y-axis.
        """
        # Create tensor representing indices along the y-axis
        p_0_x = torch.arange(1, h * self.stride + 1, self.stride)
        p_0_x = torch.flatten(p_0_x).view(1, h).repeat(1, N, 1)
        p_0 = p_0_x.type(dtype)
        p_0.requires_grad = False

        return p_0

    def _get_p(self, offset, dtype):
        """
        Calculate the deformation tensor.

        Parameters:
            offset (Tensor): Offset tensor.
            dtype (torch.dtype): Data type of the tensor.

        Returns:
            Tensor: Deformation tensor.
        """
        N, h = offset.size(1), offset.size(2)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, N, dtype)

        # Calculate the deformation tensor
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        """
        Extract values from the input tensor based on indices.

        Parameters:
            x (Tensor): Input tensor.
            q (Tensor): Tensor representing indices.

        Returns:
            Tensor: Extracted values from the input tensor.
        """
        b, h, _ = q.size()
        padded_w = x.size(2)
        c = x.size(1)

        # Reshape the input tensor for indexing
        x = x.contiguous().view(b, c, -1)
        index = q * padded_w
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1).contiguous().view(b, c, -1)

        # Gather values from the input tensor based on indices
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        """
        Reshape the x_offset tensor.

        Parameters:
            x_offset (Tensor): Input tensor.
            ks (int): Kernel size.

        Returns:
            Tensor: Reshaped tensor.
        """
        b, c, h, N = x_offset.size()

        # Reshape the x_offset tensor
        x_offset = x_offset.contiguous().view(b, c, h * ks)

        return x_offset
