import gc
import time
import torch
from torch.autograd import Function


def custom_mask_coo(tensor, mask_indices):
    # This is fine, but less efficient that doing tensor * mask_indices as if we
    #  were Neanderthals
    num_columns = torch.max(mask_indices[1]) + 1

    tensor_keys = tensor.indices()[0] * num_columns + tensor.indices()[1]
    mask_keys = mask_indices[0] * num_columns + mask_indices[1]

    mask = torch.isin(tensor_keys, mask_keys)

    return tensor.values()[mask]


def custom_mask_csr(tensor, mask_indices):
    # Same as before, less efficient than tensor * mask_indices

    # Determine the maximum number of columns in sp for creating unique keys
    num_columns = tensor.size(1)

    # Compute flat indices for each element in sp
    row_lengths = tensor.crow_indices()[1:] - tensor.crow_indices()[:-1]
    row_expansions = torch.repeat_interleave(
        torch.arange(tensor.size(0), device=tensor.device), row_lengths
    )
    flat_indices = row_expansions * num_columns + tensor.col_indices()

    # Determine if sp keys are in mask keys
    mask = torch.isin(flat_indices, mask_indices)

    # Filter 'sp' values based on mask
    return tensor.values()[mask]


class SparseMatrixMul(Function):
    @staticmethod
    def forward(ctx, indices, values, shape, layer_number, x):

        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, dtype=values.dtype, device=values.device
        )
        results = [x]
        for _ in range(layer_number):
            x = torch.sparse.mm(sparse_tensor, x)
            results.append(x)

        ctx.save_for_backward(sparse_tensor)
        ctx.intermediates = results
        ctx.layer_number = layer_number

        return results[-1]

    @staticmethod
    def backward(ctx, grad_output):
        sparse_tensor = ctx.saved_tensors[0].coalesce()
        intermediates = ctx.intermediates
        layer_number = ctx.layer_number
        device = grad_output.device

        # this is to be able to retrieve the values only in the correct positions
        sparse_tensor_non_zero = torch.sparse_coo_tensor(
            sparse_tensor.indices(), torch.ones_like(sparse_tensor.values()), sparse_tensor.shape, device=device
        )

        grad_weights = torch.zeros_like(sparse_tensor.values(), dtype=torch.float)
        sparse_tensor = sparse_tensor.t()
        for i in range(layer_number):
            print(f"Computing layer {i}")

            start_time = time.time()
            outer_product = torch.sparse.mm(
                grad_output, intermediates[layer_number - 1 - i].t()
            )
            print(f"Time to compute outer product: {time.time() - start_time}")

            if i == 0:
                full_weight_tensor = outer_product
            elif i == 1:
                full_weight_tensor = torch.sparse.mm(sparse_tensor, outer_product)
            else:
                for _ in range(i - 2):
                    sparse_tensor = torch.sparse.mm(sparse_tensor, sparse_tensor)
                full_weight_tensor = torch.sparse.mm(sparse_tensor, outer_product)
            del outer_product
            gc.collect()

            grad_weights += (
                (full_weight_tensor * sparse_tensor_non_zero).coalesce().values()
            )
            del full_weight_tensor
            gc.collect()

            print(f"Done with layer {i}")

        # Lookout, we are reusing the last sparse_tensor computed before
        grad_input = torch.sparse.mm(
            torch.sparse.mm(sparse_tensor, sparse_tensor),
            grad_output,
        )

        return None, grad_weights, None, None, grad_input


class CompressedSparseMatrixMul(Function):
    @staticmethod
    def forward(ctx, indices, values, shape, layer_number, x, sparse_layout):

        sparse_tensor = torch.sparse_compressed_tensor(
            indices, values, shape, dtype=values.dtype, device=values.device, layout=sparse_layout
        )
        results = [x]
        for _ in range(layer_number):
            x = torch.sparse.mm(sparse_tensor, x)
            results.append(x)

        ctx.save_for_backward(sparse_tensor)
        ctx.intermediates = results
        ctx.layer_number = layer_number

        return results[-1]

    @staticmethod
    def backward(ctx, grad_output):
        sparse_tensor = ctx.saved_tensors
        intermediates = ctx.intermediates
        layer_number = ctx.layer_number
        device = grad_output.device

        # this is to be able to retrieve the values only in the correct positions
        sparse_tensor_non_zero = torch.sparse_coo_tensor(
            sparse_tensor.indices(), torch.ones_like(sparse_tensor.values()), sparse_tensor.shape, device=device
        )

        grad_weights = torch.zeros_like(sparse_tensor.values(), dtype=torch.float)
        sparse_tensor = sparse_tensor.t()
        for i in range(layer_number):
            print(f"Computing layer {i}")

            start_time = time.time()
            outer_product = torch.sparse.mm(
                grad_output, intermediates[layer_number - 1 - i].t()
            )
            print(f"Time to compute outer product: {time.time() - start_time}")

            if i == 0:
                full_weight_tensor = outer_product
            elif i == 1:
                full_weight_tensor = torch.sparse.mm(sparse_tensor, outer_product)
            else:
                for _ in range(i - 2):
                    sparse_tensor = torch.sparse.mm(sparse_tensor, sparse_tensor)
                full_weight_tensor = torch.sparse.mm(sparse_tensor, outer_product)
            del outer_product
            gc.collect()

            grad_weights += (
                (full_weight_tensor * sparse_tensor_non_zero).coalesce().values()
            )
            del full_weight_tensor
            gc.collect()

            print(f"Done with layer {i}")

        # Lookout, we are reusing the last sparse_tensor computed before
        grad_input = torch.sparse.mm(
            torch.sparse.mm(sparse_tensor, sparse_tensor),
            grad_output,
        )

        return None, grad_weights, None, None, grad_input
