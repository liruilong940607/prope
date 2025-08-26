# PRoPE
https://www.liruilong.cn/prope/

This branch shows how to apply different camera conditioning approaches (including [our proposed PRoPE](https://www.liruilong.cn/prope/)) to a [LVSM](https://haian-jin.github.io/projects/LVSM/) model for the task of novel view synthesis.

## Setup

```
pip install -r requirements.txt 
pip install . # this will install two packages: prope, nvs
```

To make sure your setup works, you could run `pytest tests/`.

## Dataset

We first download the [RealEstate10K](https://google.github.io/realestate10k/) dataset using the script [`scripts/gen_imgs.py`](scripts/gen_imgs.py). Then we run [`scripts/gen_transforms.py`](scripts/gen_transforms.py) and [`scripts/data_processes.py`](scripts/data_preprocess.py) to convert the data into our data format.

Note we were not able to download all sequences as some of them are already invalid. We mark all sequences that we used for training and validation in the file [`assets/test_split_re10k.txt`](assets/test_split_re10k.txt) and [`assets/train_split_re10k.txt`](assets/train_split_re10k.txt) for reproducibility.

## Training

We support training with pixel-aligned camera conditioning (e.g., Plucker raymap, Naive raymap, Camray) or attention-based camera conditioning (e.g., GTA, PRoPE) or the combination of them. For example training with `PRoPE+Camray` (our recommandation) with 2 GPUs is via:

```
bash ./scripts/nvs.sh --ray_encoding camray --pos_enc prope --gpus "0,1"
```

See `bash ./scripts/nvs.sh  -h` for helper information.

## Validation

The validation on different zooming-in factors can be done via:
 
```
bash ./scripts/nvs.sh --ray_encoding camray --pos_enc prope --gpus "0,1" --test-zoom-in "1 3 5"
```

And different number of input (context) views can be done via:

```
bash ./scripts/nvs.sh --ray_encoding camray --pos_enc prope --gpus "0,1" --test-context-views "2 4 8 16"
```

Or if you simplly want to render some videos out (trajectory is predefined):

```
bash ./scripts/nvs.sh --ray_encoding camray --pos_enc prope --gpus "0,1" --test-render-video
```


