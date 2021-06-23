import torch
import numpy as np

v1 = torch.tensor([1.0, 1.0])
v2 = torch.tensor([2.0, 2.0])

v_sum = v1 + v2
v_res = (v_sum*2).sum()
print(
    v_res,
    v1.is_leaf,
    v2.is_leaf,
    v_sum.is_leaf,
    v_res.is_leaf,
    v1.requires_grad,
    v2.requires_grad,
    v_res.requires_grad,
    v_sum.requires_grad,
    v1.grad,
    v2.grad
    )

v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v_sum = v1 + v2
v_res = (v_sum*2).sum()
v_res.backward()
print(
    v_res,
    v1.is_leaf,
    v2.is_leaf,
    v_sum.is_leaf,
    v_res.is_leaf,
    v1.requires_grad,
    v2.requires_grad,
    v_res.requires_grad,
    v_sum.requires_grad,
    v1.grad,
    v2.grad
    )