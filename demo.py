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

import torch
import torch.nn.functional as F

from prope.per_token_camera import PropeDotProductAttention
from prope.utils.functional import random_SE3, random_SO3


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

    # Camera parameters for each token.
    viewmats = random_SE3(batch_size=(batch, seqlen))  # (batch, seqlen, 4, 4)
    Ks = random_SO3(batch_size=(batch, seqlen))  # (batch, seqlen, 3, 3)

    # Assuming there are some tokens do not have camera information (e.g., language tokens),
    # It does not matter if they are before or after the camera tokens. We will split them out
    # and only apply PRoPE to the camera tokens.
    seqlen_other_modalities = 10

    # Q/K/V right before the F.scaled_dot_product_attention. Assume multimodal.
    q = torch.randn(batch, num_heads, seqlen + seqlen_other_modalities, head_dim)
    k = torch.randn(batch, num_heads, seqlen + seqlen_other_modalities, head_dim)
    v = torch.randn(batch, num_heads, seqlen + seqlen_other_modalities, head_dim)

    # Apply F.scaled_dot_product_attention with PRoPE camera encoding.
    prope = PropeDotProductAttention(
        head_dim=head_dim,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )
    prope._precompute_and_cache_apply_fns(viewmats, Ks)

    # Apply PRoPE to *only* the camera tokens. Tokens without camera information are not affected.
    kwargs = {}
    q = torch.cat([prope._apply_to_q(q[:, :, :seqlen, :]), q[:, :, seqlen:, :]], dim=2)
    k = torch.cat([prope._apply_to_kv(k[:, :, :seqlen, :]), k[:, :, seqlen:, :]], dim=2)
    v = torch.cat([prope._apply_to_kv(v[:, :, :seqlen, :]), v[:, :, seqlen:, :]], dim=2)
    # Here you could add pass in other args like dropout etc.
    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
    o = torch.cat([prope._apply_to_o(o[:, :, :seqlen, :]), o[:, :, seqlen:, :]], dim=2)


if __name__ == "__main__":
    demo()
