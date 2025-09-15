# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Tuple

import torch

from prope.per_token_camera import PropeDotProductAttention
from prope.utils.functional import random_SE3, random_SO3


def random_SO3(batch_size: Tuple[int], device="cpu"):
    # Step 1: Generate a batch of random matrices of shape (batch_size, 3, 3)
    random_matrices = torch.randn((*batch_size, 3, 3), device=device)
    random_matrices = random_matrices.reshape(-1, 3, 3)

    # Step 2: Apply QR decomposition to each matrix in the batch
    # The `torch.linalg.qr` function works for batches of matrices in newer PyTorch versions
    q, r = torch.linalg.qr(random_matrices)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]

    # Step 3: Adjust for positive determinant in each matrix
    # Compute the determinants and find indices where the determinant is negative
    det_q = torch.det(q)
    negative_det_indices = det_q < 0

    # Flip the sign of the last column where determinant is negative
    q[negative_det_indices, :, 2] *= -1
    q = q.reshape(*batch_size, 3, 3)

    return q


def random_SE3(batch_size: Tuple[int], device="cpu"):
    random_matrices = torch.eye(4, device=device).repeat(*batch_size, 1, 1)
    random_matrices[..., :3, :3] = random_SO3(batch_size, device)
    random_matrices[..., :3, 3] = torch.randn(*batch_size, 3, device=device)
    return random_matrices


def demo():
    torch.manual_seed(42)

    cameras = 3  # How many cameras are concatenated along the seqlen axis?
    patches_x = 8  # How many patches wide is each image?
    patches_y = 8  # How many patches tall is each image?
    image_width = 128  # Width of the image. Used to normalize intrinsics.
    image_height = 128  # Height of the image. Used to normalize intrinsics.

    batch = 2
    seqlen = (
        cameras * patches_x * patches_y
    )  # Total number of tokens across multiview images.
    num_heads = 4
    head_dim = 16

    # Some tokens before the attention.
    q = torch.randn(batch, num_heads, seqlen, head_dim)
    k = torch.randn(batch, num_heads, seqlen, head_dim)
    v = torch.randn(batch, num_heads, seqlen, head_dim)

    # Camera parameters for each token.
    # Note: if there are some tokens do not have camera information (e.g., language tokens),
    # you can set the corresponding viewmats and Ks to identity matrices for those tokens.
    viewmats = random_SE3(batch_size=(batch, seqlen))  # (batch, seqlen, 4, 4)
    Ks = random_SO3(batch_size=(batch, seqlen))  # (batch, seqlen, 3, 3)

    # Apply F.scaled_dot_product_attention with PRoPE camera encoding.
    prope = PropeDotProductAttention(
        head_dim=head_dim,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    o = prope(q, k, v, viewmats, Ks)  # (batch, num_heads, seqlen, head_dim)


if __name__ == "__main__":
    demo()
