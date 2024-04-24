import gc
import torch
from torch.autograd import Function


def create_sparse_diagonal_matrix(size, dtype=torch.float32, device="cpu"):
    """
    Creates a sparse diagonal matrix (identity matrix) of the given size.

    Parameters:
    - size (int): The size of the NxN diagonal matrix.
    - dtype (torch.dtype): The data type of the matrix elements.
    - device (str or torch.device): The device on which to store the matrix.

    Returns:
    - torch.Tensor: A sparse diagonal matrix of shape (size, size).
    """
    indices = torch.arange(size, dtype=dtype, device=device).unsqueeze(0)
    indices = torch.cat(
        [indices, indices], dim=0
    )  # Make it 2D for COO format (rows and cols)
    values = torch.ones(size, dtype=dtype, device=device) 

    # Create the sparse COO tensor
    return torch.sparse_coo_tensor(
        indices, values, (size, size), dtype=dtype, device=device
    )


def sparse_outer_product(sp1, sp2, dtype=torch.float32):
    """
    Computes a sparse "outer product" of two sparse tensors, where sp1 is an N x 1 column vector
    and sp2 is an M x 1 column vector, and returns a sparse tensor representing the outer product.

    Parameters:
    sp1 (torch.Tensor): Sparse tensor of size N x 1.
    sp2 (torch.Tensor): Sparse tensor of size M x 1.

    Returns:
    torch.Tensor: Resulting sparse tensor after performing the "outer product".
    """
    # Ensure sp1 and sp2 are coalesced
    sp1 = sp1.coalesce()
    sp2 = sp2.coalesce()

    # Since sp2 is M x 1, we need to convert it to a 1 x M row vector
    indices_sp2 = sp2.indices()[0]
    size_sp2 = (1, sp2.size(0))
    sp2_row = torch.sparse_coo_tensor(
        torch.stack(
            [torch.zeros_like(indices_sp2), indices_sp2]
        ),
        sp2.values(),
        size_sp2,
        dtype=dtype,
    )

    # Perform sparse matrix multiplication to compute the outer product
    return torch.sparse.mm(sp1, sp2_row)


class SparseMatrixMulFunction(Function):
    @staticmethod
    def forward(ctx, indices, values, shape, layer_number, x):

        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, dtype=values.dtype, device=values.device
        )
        results = [x]
        for _ in range(layer_number):
            x = torch.sparse.mm(sparse_tensor, x)
            results.append(x)

        ctx.save_for_backward(indices, values, sparse_tensor)
        ctx.intermediates = results
        ctx.layer_number = layer_number

        return results[-1]

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights, sparse_tensor = ctx.saved_tensors
        intermediates = ctx.intermediates
        layer_number = ctx.layer_number
        device = grad_output.device

        grad_output  = grad_output.float()
        intermediates = [a.float() for a in intermediates]
        sparse_tensor = sparse_tensor.float()

        # this is to be able to retrieve the values only in the correct positionsn
        sparse_tensor_non_zero = torch.sparse_coo_tensor(
            indices, torch.ones_like(weights), sparse_tensor.shape, device=device
        )

        grad_weights = torch.zeros_like(weights, dtype=torch.float)
        sparse_tensor = sparse_tensor.t()
        for i in range(layer_number):
            print(f"Computing layer {i}")

            outer_product = sparse_outer_product(
                grad_output, intermediates[layer_number - 1 - i], dtype=torch.float
            )

            print("i'm before the if")

            if i == 0:
                full_weight_tensor = outer_product
            elif i == 1:
                print("alive")
                full_weight_tensor = torch.sparse.mm(
                    sparse_tensor, outer_product
                )
            else:
                for _ in range(i - 2):
                    sparse_tensor = torch.sparse.mm(sparse_tensor, sparse_tensor)
                full_weight_tensor = torch.sparse.mm(sparse_tensor, outer_product)
            del outer_product

            print(f"I'm before the grad_weights +=")
            grad_weights += (
                (full_weight_tensor * sparse_tensor_non_zero).coalesce().values()
            )
            del full_weight_tensor
            # garbage collect
            gc.collect()

            print(f"Done with layer {i}")

        # Lookout, we are reusing the last sparse_tensor computed before
        grad_input = torch.sparse.mm(
            torch.sparse.mm(sparse_tensor, sparse_tensor),
            grad_output,
        )

        return None, grad_weights, None, None, grad_input
