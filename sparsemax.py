#From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
from typing import Optional, Tuple
import torch
import torch.nn as nn

def flatten_all_but_nth_dim(ctx, x: torch.Tensor):
    """
    Flattens tensor in all but 1 chosen dimension.
    Saves necessary context for backward pass and unflattening.
    """

    # transpose batch and nth dim
    if ctx.dim != 0:
        x = x.transpose(0, ctx.dim)
    # x = x.transpose(0, ctx.dim)

    # Get and save original size in context for backward pass
    original_size = x.size()
    ctx.original_size = original_size

    # Flatten all dimensions except nth dim
    x = x.reshape(x.size(0), -1)

    # Transpose flattened dimensions to 0th dim, nth dim to last dim
    return ctx, x.transpose(0, -1)

def unflatten_all_but_nth_dim(ctx, x: torch.Tensor):
    """
    Unflattens tensor using necessary context
    """
    # Tranpose flattened dim to last dim, nth dim to 0th dim
    x = x.transpose(0, 1)

    # Reshape to original size
    x = x.reshape(ctx.original_size)

    # Swap batch dim and nth dim
    if ctx.dim != 0:
        x = x.transpose(0, ctx.dim)
    # return ctx, x.transpose(0, ctx.dim)
    return ctx, x

class SparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1):
        input_dim = input.dim()
        if input_dim <= dim or dim < -input_dim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{input_dim}, {input_dim - 1}], but got {dim})"
            )
        if dim < 0:
            dim = input_dim + dim

        # Save operating dimension to context
        ctx.needs_reshaping = input_dim > 2
        ctx.dim = dim
        ctx.tranpose = (dim == 0) and (input_dim == 2)

        if ctx.needs_reshaping:
            ctx, input = flatten_all_but_nth_dim(ctx, input)
        elif ctx.tranpose:
            input = torch.transpose(input, 0, 1)

        device = input.device
        dtype = input.dtype
        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        zs = input.sort(-1, descending=True).values
        range = torch.arange(1, input.size()[-1] + 1, dtype=dtype, device=device)
        range = range.expand_as(input)#.to(input)

        # Determine sparsity of projection
        bound = 1 + range * zs
        is_gt = bound.gt(zs.cumsum(-1)).type(input.dtype)
        k = (is_gt * range).max(-1, keepdim=True).values

        # Compute threshold
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (zs_sparse.sum(-1, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)

        # Save context
        ctx.save_for_backward(output)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, output = unflatten_all_but_nth_dim(ctx, output)
        elif ctx.tranpose:
            output = torch.transpose(output, 0, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, *_ = ctx.saved_tensors

        # Reshape if needed
        if ctx.needs_reshaping:
            ctx, grad_output = flatten_all_but_nth_dim(ctx, grad_output)
        elif ctx.tranpose:
            grad_output = torch.transpose(grad_output, 0, 1)

        # Compute gradient
        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(-1, keepdim=True)
        sum = (grad_output * nonzeros).sum(-1, keepdim=True) / num_nonzeros
        grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, grad_input = unflatten_all_but_nth_dim(ctx, grad_input)
        elif ctx.tranpose:
            grad_input = torch.transpose(grad_input, 0, 1)

        return grad_input, None
    
def sparsemax(x: torch.Tensor, dim: int = -1):
    return SparsemaxFunction.apply(x, dim)

class Sparsemax(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1):
        """
        Sparsemax class as seen in https://arxiv.org/pdf/1602.02068.pdf
        Parameters
        ----------
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input):
        # return SparsemaxFunction.apply(input, self.dim)
        return sparsemax(input, dim=self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class SingleLabelSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input.shape = (batch_size, C)

            target.shape = (batch_size, )

            weight.shape = (C, )
        Returns:
            .shape = (batch_size, )
        """
        batch_size, C = input.size()
        device = input.device
        dtype = input.dtype
        batch_idx = torch.arange(0, batch_size, dtype=torch.long, device=device)
        # C_idx = torch.arange(0, C, dtype=torch.long, device=torch.device)
        C_idx = target

        #z_k.shape = (batch_size, )
        z_k = input[batch_idx, C_idx]
        weight_k = None
        if weight is not None:
            #weight0.shape = (1, C)
            #weight1.shape = (batch_size, C)
            #weight_k.shape = (batch_size, )
            weight0 = torch.unsqueeze(weight, 0)
            weight1 = weight0.expand_as(input)
            weight_k = weight1[batch_idx, C_idx]



        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        #zs.shape = (batch_size, C)
        zs = input.sort(-1, descending=True).values

        #range.shape = (1, C)
        #range.shape = (batch_size, C)
        range = torch.unsqueeze(torch.arange(1, C + 1, dtype=dtype, device=device), 0)
        range = range.expand_as(input)

        # Determine sparsity of projection
        #bound.shape = (batch_size, C)
        #is_gt.shape = (batch_size, C)
        #k.shape = (batch_size, 1)
        bound = 1.0 + range * zs
        is_gt = bound.gt(zs.cumsum(-1)).type(dtype)
        k = (is_gt * range).max(-1, keepdim=True).values

        # Compute threshold
        #zs_spare.shape = (batch_size, C)
        zs_sparse = is_gt * zs

        # Compute taus
        #taus.shape = (batch_size, 1)
        # taus.shape = (batch_size, C)
        taus = (zs_sparse.sum(-1, keepdim=True) - 1) / k
        # taus = taus.expand_as(input)

        #z2.shape = (batch_size, C)
        #t2.shape = (batch_size, 1)
        #z2_t2.shape = (batch_size, C)
        #z_t2.shape = (batch_size, )
        z2 = torch.square(input)
        t2 = torch.square(taus)
        z2_t2 = z2 - t2
        z_t2 = torch.sum(torch.max(torch.zeros_like(input), z2_t2), dim=1)

        #loss.shape = (batch_size, )
        original_loss = 0.5*(z_t2 + 1.0) - z_k
        if weight_k is not None:
            loss = original_loss*weight_k
        else:
            loss = original_loss
        ctx.save_for_backward(input, target, weight_k, taus, original_loss)
        return loss

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grad.shape = (batch_size, )
        Returns:
            grad_input.shape = (batch_size, C)
            grad_target = None
            grad_weight = None if weight is None
            grad_weight.shape = (C, ) if weight is not None
        """
        input, target, weight_k, taus, original_loss = ctx.saved_tensors
        dtype = input.dtype
        device = input.device
        if weight_k is None:
            #grad_original_loss.shape = (batch_size, )
            grad_weight = None
            grad_original_loss = grad

            #sparsemax_v.shape = (batch_size, C)
            #grad_index.shape = (batch_size, 1)
            #grad_src.shape = (batch_size, 1)
            sparsemax_v = torch.max(torch.zeros_like(input, dtype=dtype, device=device), input - taus)
            grad_index = torch.unsqueeze(target, 1)
            grad_src = -torch.ones_like(grad_index, dtype=dtype, device=device)
            # grad_src = -torch.unsqueeze(grad_original_loss, 1)

            #grad_input0.shape = (batch_size, C)
            #grad_original_loss_t.shape = (batch_size, 1)
            grad_input0 = sparsemax_v.scatter_add_(dim=1, index=grad_index, src=grad_src)
            grad_original_loss_t = torch.unsqueeze(grad_original_loss, 1)

            #grad_input.shape = (batch_size, C)
            grad_input = grad_input0 * grad_original_loss_t
        else:
            #grad_weight1.shape = (batch_size, )
            #grad_origial_weight.shape = (batch_size, )
            grad_weight1 = grad * original_loss
            grad_original_loss = grad * weight_k

            #grad_src.shape = (batch_size, 1)
            #grad_index.shape = (batch_size, 1)
            grad_src = torch.unsqueeze(grad_weight1, 1)
            grad_index = torch.unsqueeze(target, 1)

            #grad_weight2.shape = (batch_size, C)
            #grad_weight3.shape = (batch_size, C)
            #grad_weight.shape = (C, )
            grad_weight2 = torch.zeros_like(input, dtype=input.dtype, device=device)
            grad_weight3 = grad_weight2.scatter_(dim=1, index=grad_index, src=grad_src)
            grad_weight = torch.sum(grad_weight3, dim=0)


            sparsemax_v = torch.max(torch.zeros_like(input, dtype=dtype, device=device), input - taus)
            # grad_index = torch.unsqueeze(target, 1)
            grad_src = -torch.ones_like(grad_index, dtype=dtype, device=device)
            # grad_src = -torch.unsqueeze(grad_original_loss, 1)

            #grad_input0.shape = (batch_size, C)
            #grad_original_loss_t.shape = (batch_size, 1)
            grad_input0 = sparsemax_v.scatter_add_(dim=1, index=grad_index, src=grad_src)
            grad_original_loss_t = torch.unsqueeze(grad_original_loss, 1)

            #grad_input.shape = (batch_size, C)
            grad_input = grad_input0 * grad_original_loss_t


        grad_target = None
        return grad_input, grad_target, grad_weight

def single_label_sparsemax(
    input: torch.Tensor, 
    target: torch.Tensor, 
    weight: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Args:
        input.shape = (batch_size, C)

        target.shape = (batch_size, )

        weight.shape = (C, )

        reduction in ['mean', 'sum', 'none']

    Returns:
        .shape = (batch_size, ) if reduction is 'none'

        .shape = (, ) if reduction is 'mean' or 'sum'
    """

    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f'reduction is {reduction} not in [`mean`, `sum`, `none`]')
    
    #loss.shape = (batch_size, )
    loss = SingleLabelSparsemaxFunction.apply(input, target, weight)
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss

class SingleLabelSparsemaxLoss(nn.Module):
    def __init__(self, 
        weight: Optional[torch.Tensor] = None, 
        reduction: str = 'mean'
    ) -> None:
        """
        Args:
            weight.shape = (C, )

            reduction in ['mean', 'sum', 'none']
        """
        super(SingleLabelSparsemaxLoss, self).__init__()
        self.weight = None
        if weight is not None:
            self.weight = nn.parameter.Parameter(weight, requires_grad=False)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input.shape = (batch_size, C)

            target.shape = (batch_size, )
        Returns:
            .shape = (batch_size, ) if reduction is 'none'

            .shape = (, ) if reduction is 'sum' or 'mean'
        """
        return single_label_sparsemax(input, target, weight=self.weight, reduction=self.reduction)


class MultiLabelSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input.shape = (batch_size, C)

            target.shape = (batch_size, C)

            weight.shape = (C, )

        Returns:
            .shape = (batch_size, )
        """
        batch_size, C = input.size()
        device = input.device
        dtype = input.dtype


        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        #zs.shape = (batch_size, C)
        zs = input.sort(-1, descending=True).values

        #range.shape = (1, C)
        #range.shape = (batch_size, C)
        range = torch.unsqueeze(torch.arange(1, C + 1, dtype=dtype, device=device), 0)
        range = range.expand_as(input)

        # Determine sparsity of projection
        #bound.shape = (batch_size, C)
        #is_gt.shape = (batch_size, C)
        #k.shape = (batch_size, 1)
        bound = 1.0 + range * zs
        is_gt = bound.gt(zs.cumsum(-1)).type(dtype)
        k = (is_gt * range).max(-1, keepdim=True).values

        # Compute threshold
        #zs_spare.shape = (batch_size, C)
        zs_sparse = is_gt * zs

        # Compute taus
        #taus.shape = (batch_size, 1)
        taus = (zs_sparse.sum(-1, keepdim=True) - 1) / k

        #z2.shape = (batch_size, C)
        #t2.shape = (batch_size, 1)
        #z2_t2.shape = (batch_size, C)
        #z_t2.shape = (batch_size, C)
        #q2.shape = (batch_size, C)
        #qz.shape = (batch_size, C)
        z2 = torch.square(input)
        t2 = torch.square(taus)
        z2_t2 = z2 - t2
        z_t2 = torch.max(torch.zeros_like(input), z2_t2)
        q2 = torch.square(target)
        qz = target*input

        #L_zq.shape = (batch_size, C)
        #loss.shape = (batch_size, )
        L_zq = 0.5*(z_t2 + q2) - qz
        if weight is not None:
            L_zq = L_zq * weight
        loss = torch.sum(L_zq, dim=1)
        ctx.save_for_backward(input, target, weight, taus, L_zq)
        return loss

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grad.shape = (batch_size, )

        Returns:
            grad_input.shape = (batch_size, C)

            grad_target = (batch_size, C)

            grad_weight = None if weight is None

            grad_weight.shape = (C, ) if weight is not None
        """
        #grad_unsqueeze.shape = (batch_size, 1)
        grad_unsqueeze = torch.unsqueeze(grad, 1)

        input, target, weight, taus, L_zq = ctx.saved_tensors
        dtype = input.dtype
        device = input.device
        #sparsemax_v.shape = (batch_size, C)
        #grad_input0.shape = (batch_size, C)
        sparsemax_v = torch.max(torch.zeros_like(input, dtype=dtype, device=device), input - taus)
        grad_input0 = sparsemax_v - target
        if weight is None:
            #grad_L_zq.shape = (batch_size, C)
            #grad_input.shape = (batch_size, C)
            grad_weight = None
            grad_L_zq = grad_unsqueeze.expand_as(input)

            #grad_input.shape = (batch_size, C)
            grad_input = grad_L_zq * grad_input0


            #grad_target.shape = (batch_size, C)
            grad_target = grad_L_zq * (target - input)
        else:
            #grad_L_zq.shape = (batch_size, C)
            #grad_weight0.shape = (batch_size, C)
            #grad_weight.shape = (C, )
            grad_L_zq = grad_unsqueeze * weight
            grad_weight0 = grad_unsqueeze * L_zq
            grad_weight = torch.sum(grad_weight0, dim=0)

            #grad_input.shape = (batch_size, C)
            grad_input = grad_L_zq * grad_input0

            #grad_target.shape = (batch_size, C)
            grad_target = grad_L_zq * (target - input)

        return grad_input, grad_target, grad_weight

def multi_label_sparsemax(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
):
    """
    Args:
        input.shape = (batch_size, C)

        target.shape = (batch_size, C)

        weight.shape = (C, )

        reduction in ['mean', 'sum', 'none']
    Returns:
        .shape = (batch_size, ) if reduction is 'none'

        .shape = (, ) if reduction is 'mean' or 'sum'
    """

    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f'reduction is {reduction} not in [`mean`, `sum`, `none`]')
    
    #loss.shape = (batch_size, )
    loss = MultiLabelSparsemaxFunction.apply(input, target, weight)
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss

class MultiLabelSparsemaxLoss(nn.Module):
    def __init__(self, 
        weight: Optional[torch.Tensor] = None, 
        reduction: str = 'mean'
    ) -> None:
        """
        Args:
            weight.shape = (C, )

            reduction in ['mean', 'sum', 'none']
        """
        super(MultiLabelSparsemaxLoss, self).__init__()
        self.weight = None
        if weight is not None:
            self.weight = nn.parameter.Parameter(weight, requires_grad=False)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input.shape = (batch_size, C)

            target.shape = (batch_size, )
        Returns:
            .shape = (batch_size, ) if reduction is 'none'

            .shape = (, ) if reduction is 'sum' or 'mean'
        """
        return multi_label_sparsemax(input, target, weight=self.weight, reduction=self.reduction)



if __name__ == '__main__':
    # a = torch.randn((2,3,4))
    # print(a)
    # v = sparsemax(a, dim=2)
    # for i in range(2):
    #     for j in range(3):
    #         print(v[i, j, :])


    # a = torch.randn((3, ))
    # print(a)
    # v = sparsemax(a, dim=0)
    # print(v)

    # a = torch.randn((3, 4))
    # print(a)
    # v = sparsemax(a, dim=1)
    # v2 = torch.softmax(a, dim=1)
    # for i in range(3):
    #     print(v[i, :])
    #     print(v2[i, :])
    #     print()
    # import torch.nn.functional as F
    # a = torch.tensor([[0.0, -1.0, 2.0], [-1.0, 1.0, 2.0], [0.5, 0.5, 0.5]], requires_grad=True)
    # b = torch.tensor([2, 1, 0])
    # c = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

    # v = single_label_sparsemax(a, b, reduction='sum', weight=c)
    # # v = F.cross_entropy(a, b, reduction='sum', weight=c)
    # print(v)
    # v.backward()
    # print(a.grad)
    # print(c.grad)
    # print(sparsemax(a, dim=1))
    # # print(torch.softmax(a, dim=1))


    import torch.nn.functional as F
    a = torch.tensor([[0.0, -1.0, 2.0], [-1.0, 1.0, 2.0], [0.5, 0.5, 0.5]], requires_grad=True)
    # b = torch.tensor([2, 1, 0])
    b = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    c = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    # c = None

    v = multi_label_sparsemax(a, b, reduction='mean', weight=c)
    # v = F.cross_entropy(a, b, reduction='sum', weight=c)
    print(v)
    v.backward()
    print(a.grad)
    # print(c.grad)
    print(sparsemax(a, dim=1))
    # print(torch.softmax(a, dim=1))