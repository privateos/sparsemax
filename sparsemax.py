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


        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        zs = input.sort(-1, descending=True).values
        range = torch.arange(1, input.size()[-1] + 1)
        range = range.expand_as(input).to(input)

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

    a = torch.randn((3, 4))
    print(a)
    v = sparsemax(a, dim=1)
    for i in range(3):
        print(v[i, :])