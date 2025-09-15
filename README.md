# PRoPE
https://www.liruilong.cn/prope/

This is the official repo for the paper "Cameras as Relative Positional Encoding"

<img width="1876" height="596" alt="image" src="https://github.com/user-attachments/assets/9eba5518-b664-4d54-826c-6f35d7c84698" />

**TL;DR**: Language models and multi-view transformers must both bind “positional” information to input tokens, in terms of sequence position for LLMs and camera parameters for multi-view transformers. We present a study on camera conditioning that includes absolute positional encodings (e.g, raymaps), relative pose encodings (e.g., GTA), and a new method (PRoPE) uses *relative projective* transformation to capture 3D relationship between image tokens.

## Implementations

The implementation of PRoPE is extremely simple and efficient. We provide standalone, single-file implementations in:

- [`prope/per_img_camera.py`](prope/per_img_camera.py): All tokens in the same image have the same camera parameter. [i.e. simple pinhole camera]
- [`prope/per_token_camera.py`](prope/per_token_camera.py): Each token has its own camera parameter. This allows distorted camera by treating each patch (token) as a small pinhole camera.

## Example of Usages

See [`demo.py`](demo.py) for the case where each token has its own camera parameter.
