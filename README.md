# PRoPE
https://www.liruilong.cn/prope/

This is the official repo for the paper "Cameras as Relative Positional Encoding", published at NeurIPS 2025.

<img width="1876" height="596" alt="image" src="https://github.com/user-attachments/assets/9eba5518-b664-4d54-826c-6f35d7c84698" />

**TL;DR**: Language models and multi-view transformers must both bind “positional” information to input tokens, in terms of sequence position for LLMs and camera parameters for multi-view transformers. We present a study on camera conditioning that includes absolute positional encodings (e.g, raymaps), relative pose encodings (e.g., GTA), and a new method (PRoPE) uses *relative projective* transformation to capture 3D relationship between image tokens.

## Used by
- :fire: Tencent Hunyuan World Model 1.5 (WorldPlay) uses PRoPE for camera control: [Link](https://3d-models.hunyuan.tencent.com/world/)
- This paper extends PRoPE to work with distorted cameras: [Link](https://arxiv.org/abs/2512.07237)
- This paper uses PRoPE for feed-forward reconstruction: [Link](https://gynjn.github.io/MVP/)


## Implementations

The implementation of PRoPE is extremely simple and efficient. We provide standalone, single-file implementations for both JAX and PyTorch in [`prope/jax.py`](prope/jax.py) and [`prope/torch.py`](prope/torch.py). 

## Example of Usages

Here we demo with PyTorch version:

```python
# Say we have C images, each carries with camera infomation, which would be used for cross-view understanding.
viewmats: Tensor # (B, C, 4, 4) camera world-to-camera matrix
Ks: Tensor # (B, C, 3, 3) camera intrinsic matrix

# In transformer we typically patchify the images into tokens. Say
# the image size is (256, 384) and patch size is 16.
image_width, image_height = 256, 384
patches_x, patches_y = image_width / 16, image_height / 16

# And our attention layer has mapped the images from pixels (B, C, 384, 256) to Q/K/V tokens with shape (B, num_heads, seqlen, head_dim), where `seqlen = C * patches_x * patches_y`
Q, K, V: Tensor = ... # (B, num_heads, seqlen, head_dim)

# Injecting the camera information is simply replacing the native torch attention with our impl:
output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
# -->
output = prope_dot_product_attention(
    Q, K, V,viewmats=viewmats, Ks=Ks, patches_x=patches_x, patches_y=patches_y, image_width=image_width, image_height=image_height
)
```

## Experiments

- Improve LVSM on the task of Novel View Syntheis: [Checkout `nvs` branch](https://github.com/liruilong940607/prope/tree/nvs)
- Improve UniMatch on the task of Stereo Depth Estimation: To be released
