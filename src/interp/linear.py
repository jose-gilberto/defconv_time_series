import torch


def linear_interpolation(
    x: torch.Tensor,
    offsets: torch.Tensor,
    kernel_size: int,
    dilation: int,
    stride: int,
    dilated_positions = None,
    device: str = 'cpu',
    unconstrained: bool = False,
    _test = False,
) -> None:
    
    assert x.device == offsets.device, "x and offsets must be on same device"
    kernel_rfield=dilation*(kernel_size-1)+1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield-1,kernel_size,device=offsets.device,dtype=offsets.dtype) # kernel_size

    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2],device=offsets.device,dtype=offsets.dtype).unsqueeze(-1) # out_length x 1
    dilated_offsets_repeated = dilated_positions+offsets
    
    T = t0s + dilated_offsets_repeated # batch_size x channels x out_length x kernel_size
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s+torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    if _test:
        print("x:",x.shape) # batch_size x in_channels x input_length
        print("offsets:",offsets.shape) # batch_size x groups x out_length x kernel_size
        print("max_t0:", max_t0)
        print("t0s:",t0s.shape) # out_lengths x 1
        print("dilated positions:",dilated_positions.shape) # kernel_size
        print("dilated_offsets_repeated:",dilated_offsets_repeated.shape)
        print("T:",T.shape) # batch_size x groups x out_length x kernel_rfield

    with torch.no_grad():
        U = torch.floor(T).to(torch.long) # 1 x 1 x length x kernel_rfield
        U = torch.clamp(U,min=0,max=x.shape[2]-2)

        if _test:
            print("U:",U.shape)

        U = torch.stack([U,U+1],dim=-1)
        if U.shape[1] < x.shape[1]:
            U=U.repeat(1,x.shape[1],1,1,1)
        if _test:
            print("U:", U.shape)

    x=x.unsqueeze(-1).repeat(1,1,1,U.shape[-1])
    x = torch.stack([x.gather(index=U[:,:,:,i,:],dim=-2) for i in range(U.shape[-2])],dim=-1)
    
    G = torch.max(torch.zeros(U.shape,device=device), 1-torch.abs(U-T.unsqueeze(-1))) # batch_size x groups x out_length x kernel_rfield x kernel_size
    
    if _test:
        print("G:",G.shape)

    mx = torch.multiply(G,x.moveaxis(-2,-1))
    
    return torch.sum(mx, axis=-1) # .float()  # batch_size x channels x output_length x kernel size

