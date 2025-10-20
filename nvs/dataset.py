import glob
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _normalize_poses(
    in_c2ws: torch.Tensor,
    scene_scale_factor: float = 1.35,
):
    """
    From: https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/data/dataset_scene.py#L54-L95
    Preprocess the poses to:
    1. translate and rotate the scene to align the average camera direction and position
    2. rescale the whole scene to a fixed scale
    """

    # Translation and Rotation
    # align coordinate system (OpenCV coordinate) to the mean camera
    # center is the average of all camera centers
    # average direction vectors are computed from all camera direction vectors (average down and forward)
    center = in_c2ws[:, :3, 3].mean(0)
    avg_forward = F.normalize(
        in_c2ws[:, :3, 2].mean(0), dim=-1
    )  # average forward direction (z of opencv camera)
    avg_down = in_c2ws[:, :3, 1].mean(0)  # average down direction (y of opencv camera)
    avg_right = F.normalize(
        torch.cross(avg_down, avg_forward, dim=-1), dim=-1
    )  # (x of opencv camera)
    avg_down = F.normalize(
        torch.cross(avg_forward, avg_right, dim=-1), dim=-1
    )  # (y of opencv camera)

    avg_pose = torch.eye(4, device=in_c2ws.device)  # average c2w matrix
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    avg_pose = torch.linalg.inv(avg_pose)  # average w2c matrix
    in_c2ws = avg_pose @ in_c2ws

    # Rescale the whole scene to a fixed scale
    scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
    scene_scale = scene_scale_factor * scene_scale

    in_c2ws[:, :3, 3] /= scene_scale
    return in_c2ws


def _normalize_poses_identity_unit_distance(
    in_c2ws: torch.Tensor,
    ref0_idx: int,
    ref1_idx: int,
):
    """
    Normalize the poses such that the ref0 camera is the identity
    and the ref1 camera is unit distance to the ref0 camera.
    """

    ref0_c2w = in_c2ws[ref0_idx]
    c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(ref0_c2w), in_c2ws)

    ref0_c2w = c2ws[ref0_idx]
    ref1_c2w = c2ws[ref1_idx]
    dist = torch.linalg.norm(ref1_c2w[:3, 3] - ref0_c2w[:3, 3])
    if dist > 1e-2:  # numerically stable
        c2ws[:, :3, 3] /= dist

    return c2ws


def resize_crop_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize and crop the image to have the smallest side equal to `patch_size`,
    while maintaining sub-pixel accuracy using a single warpAffine transformation.

    Args:
        image (np.ndarray): Input image.
        K (np.ndarray): Camera intrinsic matrix.
        patch_size (int): Target size of the smaller dimension.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resized and cropped image, updated intrinsic matrix.
    """
    h, w = image.shape[:2]
    scale = patch_size / min(h, w)

    # Compute the affine transformation matrix combining scaling and cropping
    new_w, new_h = w * scale, h * scale
    crop_x = (new_w - patch_size) / 2
    crop_y = (new_h - patch_size) / 2

    M = np.array([[scale, 0, -crop_x], [0, scale, -crop_y]], dtype=np.float32)

    # Apply affine transformation with sub-pixel accuracy
    is_downsampling = min(h, w) > patch_size
    interpolation = cv2.INTER_AREA if is_downsampling else cv2.INTER_CUBIC
    cropped_resized_image = cv2.warpAffine(
        image, M, (patch_size, patch_size), flags=interpolation
    )

    # Update intrinsic matrix K
    K_scaled = K.copy()
    K_scaled[:2, :] *= scale
    K_scaled[0, 2] -= crop_x
    K_scaled[1, 2] -= crop_y

    return cropped_resized_image, K_scaled


def center_zoom_in_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Zoom into the center of the image while maintaining sub-pixel accuracy using warpAffine.

    Args:
        image (np.ndarray): Input image.
        K (np.ndarray): Camera intrinsic matrix.
        scale (float): Zoom-in factor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Zoomed image, updated intrinsic matrix.
    """
    if scale == 1.0:
        return image, K

    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2

    # Compute the affine transformation matrix for zooming in at the center
    M = np.array(
        [[scale, 0, (1 - scale) * center_x], [0, scale, (1 - scale) * center_y]],
        dtype=np.float32,
    )

    # Apply affine transformation
    zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_AREA)

    # Update intrinsic matrix K.
    K_zoomed = K.copy()
    K_zoomed[:2, :] *= scale
    K_zoomed[0, 2] += (1 - scale) * center_x
    K_zoomed[1, 2] += (1 - scale) * center_y

    return zoomed_image, K_zoomed


def load_and_maybe_update_meta_info(json_path: str) -> Tuple[bool, Dict]:
    """Load the meta information from the `transforms.json` file.

    If the image paths (e.g., `images/xxx.jpg` ) stored in the `transforms.json` file
    are not found, try with `images_{1, 2, 4, 8}/xxx.jpg` instead, and update the
    camera intrinsics accordingly.
    """
    if not os.path.exists(json_path):
        return False, {}
    with open(json_path, "r") as f:
        meta_info = json.load(f)

    # Check if the image paths are valid
    frames = meta_info["frames"]
    if len(frames) == 0:
        return False, {}

    # Use the jpg version if it exists (for DL3DV)
    if "-jpeg" in json_path:
        for frame in frames:
            frame["file_path"] = os.path.splitext(frame["file_path"])[0] + ".jpeg"

    # Check if the image paths are valid
    maybe_relative_path_to_img = frames[0]["file_path"]
    if maybe_relative_path_to_img.startswith("/"):
        _start_path = os.path.abspath(os.path.dirname(json_path))
        for frame in frames:
            frame["file_path"] = os.path.relpath(frame["file_path"], start=_start_path)
        relative_path_to_img = frames[0]["file_path"]
    else:
        relative_path_to_img = maybe_relative_path_to_img

    abs_path_to_img = os.path.join(os.path.dirname(json_path), relative_path_to_img)
    if not os.path.exists(abs_path_to_img):
        # Try with images_{1, 2, 4, 8}/xxx.jpg
        assert (
            "images/" in relative_path_to_img
        ), f"Invalid image path in the meta info file: {relative_path_to_img}"
        factor = None
        for _factor in [1, 2, 4, 8]:
            _relative_path_to_img = relative_path_to_img.replace(
                "images/", f"images_{_factor}/"
            )
            _abs_path_to_img = os.path.join(
                os.path.dirname(json_path), _relative_path_to_img
            )
            if os.path.exists(_abs_path_to_img):
                factor = _factor
                break

        if factor is None:
            # No valid image path found
            return False, meta_info

        # Found a valid image path, update the meta info
        for frame in frames:
            frame["file_path"] = frame["file_path"].replace(
                "images/", f"images_{factor}/"
            )

        w, h = meta_info["w"], meta_info["h"]
        assert (
            w % factor == 0 and h % factor == 0
        ), f"Invalid factor: {factor} with w={w} and h={h}"

        for key in ["fl_x", "fl_y", "cx", "cy"]:
            meta_info[key] /= factor
        for key in ["w", "h"]:
            meta_info[key] //= factor

    return True, meta_info


def load_frames_from_meta_info(
    data_dir: str,
    meta_info: Dict[str, Any],
    frame_ids: List[int],
    patch_size: int = 256,
    zoom_factor: float = 1.0,
    random_zoom: bool = False,
    camera_pose_only: bool = False,
) -> Union[Dict[str, Any], np.ndarray]:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    # Load the camera intrinsic
    K_raw = np.array(
        [
            [meta_info["fl_x"], 0, meta_info["cx"]],
            [0, meta_info["fl_y"], meta_info["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    frames = meta_info["frames"]

    # Shortcut for loading only camera poses
    if camera_pose_only:
        c2ws = []
        for frame_id in frame_ids:
            frame = frames[frame_id]
            c2w = np.array(frame["transform_matrix"], dtype=np.float32) @ blender2opencv
            c2ws.append(c2w)
        return np.stack(c2ws)

    # Load the images
    images, Ks, c2ws, abs_image_paths = [], [], [], []

    for index, frame_id in enumerate(frame_ids):
        frame = frames[frame_id]

        rel_image_path = frame["file_path"]
        abs_image_path = os.path.join(data_dir, rel_image_path)
        image = imageio.imread(abs_image_path)[..., :3]

        per_image_zoom_factor = (
            np.random.uniform(1.0, zoom_factor) if random_zoom else zoom_factor
        )

        image, K = center_zoom_in_with_subpixel_accuracy(
            image,
            K_raw,
            per_image_zoom_factor,
        )
        image, K = resize_crop_with_subpixel_accuracy(image, K, patch_size)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32) @ blender2opencv

        images.append(image)
        Ks.append(K)
        c2ws.append(c2w)
        abs_image_paths.append(abs_image_path)

    return {
        "image": np.stack(images),
        "K": np.stack(Ks),
        "camtoworld": np.stack(c2ws),
        "image_path": abs_image_paths,
    }


class TrainDataset(Dataset):
    def __init__(
        self,
        data_dirs: List[str],
        patch_size: int = 256,
        zoom_factor: float = 1.0,  # 1.0 means disabled
        random_zoom: bool = False,  # only useful when zoom_factor is > 1.0
        input_views: int = 2,
        supervise_views: int = 6,
        verbose: bool = False,
    ):
        super().__init__()
        # No list/dict in the dataset, which would cause "memory leak"
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
        self.data_dirs = np.array(data_dirs).astype(np.bytes_)
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.random_zoom = random_zoom
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.verbose = verbose
        if self.verbose:
            print(f"[TrainDataset] Initialized with {len(self.data_dirs)} scenes.")

    def __len__(self):
        return len(self.data_dirs)

    def _select_views(
        self, n_frames: int, min_frame_dist: int = 25, max_frame_dist: int = 100
    ) -> Optional[List[int]]:
        # From: https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/data/dataset_scene.py#L133C1-L149C29
        if n_frames < self.input_views + self.supervise_views:
            return None
        max_frame_dist = min(n_frames - 1, max_frame_dist)
        if max_frame_dist <= min_frame_dist:
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if n_frames <= frame_dist:
            return None
        start_index = random.randint(0, n_frames - frame_dist - 1)
        end_index = start_index + frame_dist
        supervise_indices = random.sample(
            range(start_index + 1, end_index), self.supervise_views
        )
        indices = [start_index, end_index] + supervise_indices
        return indices

    def __getitem__(self, _: Any) -> Dict[str, Any]:
        # Choose a random scene
        data_dir = str(np.random.choice(self.data_dirs), encoding="utf-8")
        valid, meta_info = load_and_maybe_update_meta_info(
            os.path.join(data_dir, "transforms.json")
        )
        assert valid, f"Invalid scene: {data_dir}"

        # Select views
        n_frames = len(meta_info["frames"])
        frame_ids = self._select_views(n_frames)
        if frame_ids is None:
            return self.__getitem__(None)

        # Put the smallest frame_id to the first and the largest frame_id to the second.
        # The rest will be used as supervised views.
        frame_ids = sorted(frame_ids)
        frame_ids = [frame_ids[0], frame_ids[-1]] + frame_ids[1:-1]

        # Load frames
        loaded = load_frames_from_meta_info(
            data_dir,
            meta_info,
            frame_ids,
            patch_size=self.patch_size,
            zoom_factor=self.zoom_factor,
            random_zoom=self.random_zoom,
        )

        # Preprocess poses
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        K = torch.from_numpy(loaded["K"]).float()
        image = torch.from_numpy(loaded["image"]).float()
        image_path = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
        }


class EvalDataset(Dataset):
    def __init__(
        self,
        folder: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,  # 1.0 means disabled
        verbose: bool = False,
        first_n: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        input_views: int = 2,
        supervise_views: int = 3,
        render_video: bool = False,
        test_index_fp: Optional[str] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.render_video = render_video

        if test_index_fp is not None:
            # pre defined test index file
            json_file_name = test_index_fp

        else:
            if render_video:
                json_file_name = "evaluation_index_re10k_video.json"
            else:
                assert (
                    input_views == 2 and supervise_views == 3
                ), f"Invalid input views and supervise views for RE10K, should be 2 and 3 respectively, but got {input_views} and {supervise_views}."
                json_file_name = "evaluation_index_re10k.json"

        index_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"../assets/{json_file_name}"
        )
        assert os.path.exists(index_file), f"Index file not found: {index_file}"
        with open(index_file, "r") as f:
            index_info = json.load(f)

        scenes_in_index_info = set([k for k, v in index_info.items() if v is not None])
        scenes_on_disk = set(os.listdir(folder))
        scenes = scenes_in_index_info & scenes_on_disk
        if verbose:
            print(
                f"[EvalDataset] Found {len(scenes_in_index_info)} scenes in the index file. "
                f"Found {len(scenes_on_disk)} scenes in the folder. "
                f"Using {len(scenes)} scenes."
            )
        scenes = sorted(scenes)

        if first_n is not None:
            scenes = scenes[:first_n]

        if rank is not None and world_size is not None:
            scenes = scenes[rank::world_size]

        data_dirs = [os.path.join(folder, scene) for scene in scenes]
        self.data_dirs = np.array(data_dirs).astype(np.bytes_)
        self.contexts = np.array([index_info[scene]["context"] for scene in scenes])
        if render_video:
            # Different scene has different target views
            self.targets = [index_info[scene]["target"] for scene in scenes]
        else:
            # All scenes have the same target views. Pack them into an array to avoid memory leak.
            self.targets = np.array([index_info[scene]["target"] for scene in scenes])

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, scene_id: int) -> Dict[str, Any]:
        data_dir = str(self.data_dirs[scene_id], encoding="utf-8")
        valid, meta_info = load_and_maybe_update_meta_info(
            os.path.join(data_dir, "transforms.json")
        )
        assert valid, f"Invalid scene: {data_dir}"

        # Load context and target views together
        context_view_ids = self.contexts[scene_id][: self.input_views]
        if self.render_video:
            target_view_ids = self.targets[scene_id]
        else:
            target_view_ids = self.targets[scene_id][: self.supervise_views]
        frame_ids = np.concatenate([context_view_ids, target_view_ids])

        try:
            loaded = load_frames_from_meta_info(
                data_dir,
                meta_info,
                frame_ids,
                patch_size=self.patch_size,
                zoom_factor=self.zoom_factor,
            )
        except Exception as e:
            print(f"Error in {data_dir}: {e}. frame_ids: {frame_ids}")
            raise e

        # Preprocess poses
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        K = torch.from_numpy(loaded["K"]).float()
        image = torch.from_numpy(loaded["image"]).float()
        image_path = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
            "scene": scene_id,
        }


if __name__ == "__main__":
    # Test the dataset.
    import glob

    """
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 nvs/dataset.py
    """

    torch.distributed.init_process_group(backend="nccl")
    world_rank = int(os.environ.get("RANK", 0))

    dataset = TrainDataset(
        sorted(glob.glob("./data_processed/realestate10k/train/*")), verbose=True
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)
    data = next(iter(dataloader))
    print(data["image"].shape, data["K"].shape, data["camtoworld"].shape)

    testset = EvalDataset(folder="./data_processed/realestate10k/test/", verbose=True)
    data = testset[0]
    print(data["image"].shape)

    torch.distributed.destroy_process_group()
