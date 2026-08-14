"""LeRobot policy transforms for robodeploy ↔ OpenPI integration.

Provides input/output transforms that bridge robodeploy's observation format
to OpenPI's internal model format. Supports multiple robot types via
configurable ``camera_map`` and ``action_dim``.

Camera mapping is configurable — by default the first N cameras are
mapped to ``base_0_rgb``, ``left_wrist_0_rgb``, ``right_wrist_0_rgb``
in order.

Use :class:`openpi.training.config.LeRobotDataConfig` to create a data config
for training or inference with these transforms.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.shared.image_tools import parse_image


def make_lerobot_example() -> dict:
    """Creates a random input example for a LeRobot policy."""
    return {
        "state": np.random.rand(14).astype(np.float32),
        "images": {
            "cam_0": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class LeRobotInputs(transforms.DataTransformFn):
    """Input transform for LeRobot policies served via robodeploy.

    Handles two input formats:

    1. **Robodeploy format** (wrapped): ``{"method": "infer", "obs": {...}}``
       — auto-unwraps the ``obs`` key.
    2. **Direct format**: ``{"state": ..., "images": {...}, "prompt": "..."}``

    Expected fields inside the observation dict:

    - ``state``: flat float32 array with joint positions.
    - ``images``: dict of ``{camera_name: CHW_or_HWC_uint8}``.
    - ``prompt`` (optional): task description string.

    Outputs the canonical OpenPI model format with ``image``, ``image_mask``,
    ``state``, and optionally ``prompt`` keys.
    """

    # Explicit mapping from robodeploy camera names to model image slots.
    # Example: {"front": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}
    # When empty or a camera is not found in the map, falls back to sorted alphabetical order.
    camera_map: dict[str, str] = dataclasses.field(default_factory=dict)

    # If True, the first camera is also used for left/right wrist slots
    # when those cameras are missing (instead of zero-filling).
    broadcast_base: bool = False

    # Default prompt to inject when data does not contain a "prompt" key.
    default_prompt: str | None = None

    # Model image slots in fixed order (PI0/PI05 convention).
    MODEL_IMAGE_KEYS: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def __call__(self, data: dict) -> dict:
        # Unwrap robodeploy message envelope if present
        if "obs" in data:
            data = data["obs"]

        # State: expected flat array
        state = np.asarray(data["state"], dtype=np.float32)

        # Images: convert CHW → HWC, normalize dtype
        raw_images = data.get("images", {})
        images_dict: dict[str, np.ndarray] = {}
        for name, img in raw_images.items():
            images_dict[name] = parse_image(img)

        if not images_dict:
            raise ValueError("At least one camera image is required for LeRobot policy.")

        # Map cameras to OpenPI slots:
        # 1) Use explicit camera_map for known names
        # 2) Fall back to sorted alphabetical order for unmapped cameras
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}

        # Build reverse lookup: model_slot → robodeploy_camera_name
        slot_to_cam: dict[str, str] = {}
        unmapped_cams: list[str] = []
        for cam_name in sorted(images_dict.keys()):
            mapped = False
            for src, dst in self.camera_map.items():
                if cam_name == src:
                    slot_to_cam[dst] = cam_name
                    mapped = True
                    break
            if not mapped:
                unmapped_cams.append(cam_name)

        # Fill remaining model slots with unmapped cameras (alphabetical order)
        for slot in self.MODEL_IMAGE_KEYS:
            if slot not in slot_to_cam and unmapped_cams:
                slot_to_cam[slot] = unmapped_cams.pop(0)

        base_img = next(iter(images_dict.values()))
        for slot in self.MODEL_IMAGE_KEYS:
            if slot in slot_to_cam:
                images[slot] = images_dict[slot_to_cam[slot]]
                image_masks[slot] = np.True_
            elif self.broadcast_base:
                images[slot] = base_img
                image_masks[slot] = np.True_
            else:
                images[slot] = np.zeros_like(base_img)
                image_masks[slot] = np.False_

        result: dict = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "prompt" in data:
            result["prompt"] = data["prompt"]
        elif self.default_prompt is not None:
            result["prompt"] = self.default_prompt

        # Actions only available during training
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"])

        return result


@dataclasses.dataclass(frozen=True)
class LeRobotOutputs(transforms.DataTransformFn):
    """Output transform for LeRobot policies.

    Slices the model's full action dimension (default 32) down to the
    robot's actual action space. Only performs truncation — the model
    always outputs 32-dim actions, and this transform slices to the
    robot-specific dimension.
    """

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim])
        return {"actions": actions}
