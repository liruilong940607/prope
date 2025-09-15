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

import pytest
import torch

# Enable highest precision
torch.set_default_dtype(torch.float64)

from prope.per_img_camera import (
    PropeDotProductAttention as PerImgCameraPropeDotProductAttention,
)
from prope.per_token_camera import (
    PropeDotProductAttention as PerTokenCameraPropeDotProductAttention,
)
from prope.utils.functional import random_SE3, random_SO3


@pytest.mark.skip(reason="benchmarking")
def test_benchmark_prope_torch():
    torch.manual_seed(42)
    cameras = 3
    patches_x = 8
    patches_y = 8
    image_width = 128
    image_height = 128

    batch = 2
    seqlen = cameras * patches_x * patches_y
    num_heads = 4
    head_dim = 16

    q = torch.randn(batch, num_heads, seqlen, head_dim)
    k = torch.randn(batch, num_heads, seqlen, head_dim)
    v = torch.randn(batch, num_heads, seqlen, head_dim)

    viewmats = random_SE3(batch_size=(batch, cameras))
    Ks = random_SO3(batch_size=(batch, cameras))

    # All tokens in the same image have the same camera parameter. [i.e. no distortion]
    prope1 = PerImgCameraPropeDotProductAttention(
        head_dim=head_dim,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    outputs1 = prope1(q, k, v, viewmats, Ks)

    # Each token has its own camera parameter. Here we repeat the camera parameters for each token.
    viewmats = torch.repeat_interleave(viewmats, patches_x * patches_y, 1)
    Ks = torch.repeat_interleave(Ks, patches_x * patches_y, 1)
    prope2 = PerTokenCameraPropeDotProductAttention(
        head_dim=head_dim,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    outputs2 = prope2(q, k, v, viewmats, Ks)

    torch.testing.assert_close(outputs1, outputs2)


if __name__ == "__main__":
    test_benchmark_prope_torch()
