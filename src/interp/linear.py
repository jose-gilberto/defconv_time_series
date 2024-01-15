import torch


def linear_interpolation(
    x: torch.Tensor,
    offsets: torch.Tensor,
    kernel_size: int,
    dilation: int,
    stride: int,
    dilated_positions = None,
    device: str = 'cpu',
    unconstrained: bool = False
) -> None:
    # Ensure that the x and offsets are in the same device
    assert x.device == offsets.device, 'The tensors x and offsets must be on same device.'
    
    # Calculate the receptive field for that kernel
    kernel_rfield = dilation * (kernel_size - 1) + 1
    
    # Every index in x (input) we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(
            0,
            kernel_rfield - 1,
            kernel_size,
            device=offsets.device,
            dtype=offsets.dtype
        )
        
    max_t0 = (offsets.shape[-2] - 1) * stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2], device=offsets.device, dtype=offsets.dtype).unsqueeze(-1)
    dilated_offsets_repeated = dilated_positions + offsets
    
    T = t0s + dilated_offsets_repeated # batch_size x channels x out_length x kernel_size
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s + torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))
        
    with torch.no_grad():
        U = torch.floor(T).to(torch.long)
        U = torch.clamp(U, min=0, max=x.shape[-2] - 2)
        
        U = torch.stack([U, U + 1], dim=-1)

        if U.shape[1] < x.shape[1]:
            U = U.repeat(1, x.shape[1], 1, 1, 1)
    
    x = x.unsqueeze(-1).repeat(1, 1, 1, U.shape[-1])

    x = torch.stack([
        x.gather(index=torch.clamp(U[:, :, :, i, :], 0, x.shape[-2] - 1), dim=-2)
        for i in range(U.shape[-2])], dim=-1)
    
    # G(a, b) = max(0, 1 - |a - b|)
    G = torch.max(
        torch.zeros(U.shape, device=device),
        1 - torch.abs(U - T.unsqueeze(-1))
    )
    
    mx = torch.multiply(G, x.moveaxis(-2, -1))
    return torch.sum(mx, axis=-1)