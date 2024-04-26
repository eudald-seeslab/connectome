import torch
from torch.autograd import gradcheck
from adult_models import SparseMatrixMul


def test_sparse_matrix_mul_function():
    # Test inputs
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    shape = (3, 3)
    layer_number = 3
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    x = x.unsqueeze(1)

    # Run gradcheck
    success = gradcheck(
        SparseMatrixMul.apply,
        (indices, values, shape, layer_number, x),
        eps=1e-6,
        atol=1e-4,
    )

    assert success, "Gradcheck failed for SparseMatrixMulFunction"


test_sparse_matrix_mul_function()
