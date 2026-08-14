"""
Labelme polygon annotations → COCO JSON dataset for SAM3 training.

Input:  raw_dataset/  (frame_*.png + frame_*.json)
Output: sam3_dataset/images/ + masks/ + annotations.json (COCO format)
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# =========================
# 配置
# =========================

INPUT_DIR = Path("/home/kemove/INNOV/datasets/sam_train/episode_000000")

OUTPUT_DIR = Path("/home/kemove/INNOV/datasets/sam_datasets_v2")

IMAGE_DIR = OUTPUT_DIR / "images"
MASK_DIR = OUTPUT_DIR / "masks"
ANNOTATION_FILE = OUTPUT_DIR / "annotations.json"

LABEL_MAPPING = {
    "hand": "hand",
    "forearm": "forearm",
    "hand_and_forearm": "whole hand and forearm",
    "hair": "hair",
}

LABEL_TO_CAT_ID = {
    "hand": 1,
    "forearm": 2,
    "whole hand and forearm": 3,
    "hair": 4,
}

CAT_ID_TO_NAME = {v: k for k, v in LABEL_TO_CAT_ID.items()}


def polygon_to_bbox(points):
    """polygon [[x,y], ...] → COCO XYWH bbox [x, y, w, h]"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return [x, y, w, h]


def bbox_area(bbox):
    """COCO area = w * h"""
    return bbox[2] * bbox[3]


def polygon_to_coco_segmentation(points):
    """
    Labelme points [[x1,y1], [x2,y2], ...] → COCO segmentation [[x1,y1,x2,y2,...]]
    每个 polygon 是 [x1,y1,x2,y2,...] 格式的扁平列表
    """
    return [[round(c) for p in points for c in p]]


def convert_dataset():
    """主转换函数：Labelme → COCO JSON"""

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    coco_categories = [
        {"id": cat_id, "name": name} for name, cat_id in LABEL_TO_CAT_ID.items()
    ]

    json_files = sorted(INPUT_DIR.glob("frame_*.json"))
    print(f"Found {len(json_files)} json files")

    ann_id = 0
    mask_index = 0

    for img_id, json_file in enumerate(tqdm(json_files), start=1):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_name = data["imagePath"]
        image_path = INPUT_DIR / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        width = data["imageWidth"]
        height = data["imageHeight"]
        image_stem = Path(image_name).stem

        coco_images.append(
            {
                "id": img_id,
                "file_name": f"images/{image_name}",
                "width": width,
                "height": height,
            }
        )

        # 复制 image
        shutil.copy(image_path, IMAGE_DIR / image_name)

        for obj_id, shape in enumerate(data["shapes"]):
            label = shape["label"]

            if label not in LABEL_MAPPING:
                print(f"Skip unknown label: {label}")
                continue

            text_prompt = LABEL_MAPPING[label]
            cat_id = LABEL_TO_CAT_ID[text_prompt]
            points = shape["points"]
            bbox = polygon_to_bbox(points)
            segm = polygon_to_coco_segmentation(points)

            # 生成并保存 mask PNG（备查）
            mask = np.zeros((height, width), dtype=np.uint8)
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 1)

            mask_name = f"{image_stem}_{obj_id:03d}.png"
            mask_path = MASK_DIR / mask_name
            cv2.imwrite(str(mask_path), mask * 255)

            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": bbox_area(bbox),
                    "segmentation": segm,
                    "iscrowd": 0,
                }
            )

            ann_id += 1
            mask_index += 1

    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print("\nFinished")
    print(f"Images: {len(coco_images)}")
    print(f"Annotations (instances): {len(coco_annotations)}")
    print(f"Masks: {mask_index}")
    print(f"Categories: {[c['name'] for c in coco_categories]}")
    print(f"Saved: {ANNOTATION_FILE}")


if __name__ == "__main__":
    convert_dataset()
