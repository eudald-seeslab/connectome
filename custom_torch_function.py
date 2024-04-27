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
    def forward(ctx, sparse_tensor, layer_number, x):

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
        w = ctx.saved_tensors[0].coalesce()
        x = ctx.intermediates
        layer_num = ctx.layer_number
        device = grad_output.device

        # this is to be able to retrieve the values only in the correct positions
        sparse_tensor_non_zero = torch.sparse_coo_tensor(
            w.indices(), torch.ones_like(w.values()), w.shape, device=device
        )

        grad_weights = torch.zeros_like(w.values(), dtype=torch.float)
        w_t = w.t()
        for i in range(layer_num):
            print(f"Computing layer {i}")

            start_time = time.time()
            outer_product = torch.sparse.mm(
                grad_output, x[layer_num - 1 - i].t()
            )
            print(f"Time to compute outer product: {time.time() - start_time}")

            if i == 0:
                full_weight_tensor = outer_product
            elif i == 1:
                full_weight_tensor = torch.sparse.mm(w_t, outer_product)
            else:
                w_t = torch.sparse.mm(w_t, w_t)
                full_weight_tensor = torch.sparse.mm(w_t, outer_product)
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
            torch.sparse.mm(w, w),
            grad_output,
        )

        return None, grad_weights, None, None, grad_input


class CompressedSparseMatrixMul(Function):
    @staticmethod
    def forward(ctx, w, layer_number, x, sparse_layout):

        results = [x]
        for _ in range(layer_number):
            x = w.matmul(x)
            results.append(x)

        ctx.save_for_backward(w)
        ctx.intermediates = results
        ctx.layer_number = layer_number
        ctx.sparse_layout = sparse_layout

        return results[-1]

    @staticmethod
    def backward(ctx, grad_output):
        w = ctx.saved_tensors[0]
        x = ctx.intermediates
        layer_number = ctx.layer_number
        sparse_layout = ctx.sparse_layout
        device = grad_output.device

        # this is to be able to retrieve the values only in the correct positions
        sparse_tensor_non_zero = torch.sparse_compressed_tensor(
            w.crow_indices(), 
            w.col_indices(), 
            torch.ones_like(w.values()), 
            layout=sparse_layout, 
            device=device
        )
        # We will only use the transpose of w to some power
        # Note that we don't create new tensors each time we multiply w by itself
        #  for the sake of memory efficiency. Be careful with this.
        w_t = w.t()

        grad_weights = torch.zeros_like(w.values(), dtype=torch.float)
        for i in range(layer_number):
            print(f"Computing layer {i}")

            start_time = time.time()
            outer_product = grad_output.matmul(x[layer_number - 1 - i].t())
            print(f"Time to compute outer product: {time.time() - start_time}")

            if i == 0:
                full_weight_tensor = outer_product
            elif i == 1:
                full_weight_tensor = w_t.matmul(outer_product)
            else:
                # the sparse_tensor carries over the previous computation,
                #  so we only have to multiply it by itself once each time
                w_t = w_t.matmul(w_t)
                full_weight_tensor = w_t.matmul(outer_product)
            del outer_product
            gc.collect()

            grad_weights += (
                (full_weight_tensor * sparse_tensor_non_zero).values()
            )
            del full_weight_tensor
            gc.collect()

            print(f"Done with layer {i}")

        # Lookout, we are reusing the last sparse_tensor computed before
        grad_input = w.matmul(
            w.matmul(grad_output),
        )
        return None, grad_weights, None, None, grad_input
