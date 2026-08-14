from collections.abc import Sequence
import logging

import kornia.augmentation as K  # noqa: N812
import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    enable_shape_aug: bool = False,
    enable_color_aug: bool = False,
    enable_blur_aug: bool = False,
):
    """预处理观测数据，支持可选的训练时图像增强。"""
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3

        if is_channels_first:
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train and (enable_shape_aug or enable_color_aug or enable_blur_aug):
            image = image / 2.0 + 0.5  # [-1,1] -> [0,1]

            if enable_shape_aug and "wrist" not in key:
                height, width = image.shape[1:3]

                # 随机裁剪 (95%)
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

                # 随机旋转 ±5°
                angle = torch.rand(1, device=image.device) * 10 - 5
                if torch.abs(angle) > 0.1:
                    angle_rad = angle * torch.pi / 180.0
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)
                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

            if enable_color_aug:
                # 亮度抖动 [0.7, 1.3]
                brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
                image = image * brightness_factor
                # 对比度抖动 [0.6, 1.4]
                contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
                mean = image.mean(dim=[1, 2, 3], keepdim=True)
                image = (image - mean) * contrast_factor + mean
                # 饱和度抖动 [0.5, 1.5]
                saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0
                gray = image.mean(dim=-1, keepdim=True)
                image = gray + (image - gray) * saturation_factor

            if enable_blur_aug:
                # 中值模糊 + 运动模糊（与 RISE 一致，各 p=0.1）
                image_nchw = image.permute(0, 3, 1, 2).contiguous()
                aug = K.AugmentationSequential(
                    K.RandomMedianBlur(kernel_size=(3, 5), p=0.1),
                    K.RandomMotionBlur(kernel_size=(3, 5), angle=35.0, direction=0.5, p=0.1),
                    keepdim=True,
                )
                image_nchw = aug(image_nchw)
                image = image_nchw.permute(0, 2, 3, 1).contiguous()

            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0  # [0,1] -> [-1,1]

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
