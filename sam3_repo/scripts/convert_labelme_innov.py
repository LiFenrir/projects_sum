#!/usr/bin/env python3
"""
汇总 innov_arm 的 labelme 标注(part1 + part2)→ COCO JSON 训练集。

含空标注图(纯负样本)。输出 images/ + annotations.json,并打印类别统计。
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

# =========================
# 配置
# =========================

INPUT_DIRS = [
    Path("/home/kemove/INNOV/datasets/innov_arm/labelme_sampled"),
    Path("/home/kemove/INNOV/datasets/innov_arm/labelme_sampled_part2"),
]

OUTPUT_DIR = Path("/home/kemove/INNOV/datasets/innov_arm/innov_coco_v1")
IMAGE_DIR = OUTPUT_DIR / "images"
ANNOTATION_FILE = OUTPUT_DIR / "annotations.json"

# labelme 标签 → 训练类别名(即文本提示词)
LABEL_TO_CAT_ID = {
    "operator": 1,
    "robot arm": 2,
}


def polygon_to_bbox(points):
    """polygon [[x,y], ...] → COCO XYWH bbox"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x, y = min(xs), min(ys)
    return [x, y, max(xs) - x, max(ys) - y]


def convert_dataset():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    json_files = []
    for d in INPUT_DIRS:
        json_files += [p for p in sorted(d.glob("*.json"))
                       if p.name != "manifest.json"]
    print(f"Found {len(json_files)} labelme json files")

    coco_images = []
    coco_annotations = []
    ann_id = 0
    # 统计: 每类实例数、每类出现的图片数、空标注图数
    inst_counter = Counter()
    img_with_class = Counter()
    empty_images = 0
    skipped_labels = Counter()

    for img_id, json_file in enumerate(json_files, start=1):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        image_name = json_file.with_suffix(".png").name
        image_path = json_file.parent / image_name
        if not image_path.exists():
            print(f"Skip missing image: {image_path}")
            continue

        width, height = data["imageWidth"], data["imageHeight"]
        coco_images.append({
            "id": img_id,
            "file_name": f"images/{image_name}",
            "width": width,
            "height": height,
        })
        shutil.copy(image_path, IMAGE_DIR / image_name)

        classes_in_img = set()
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in LABEL_TO_CAT_ID:
                skipped_labels[label] += 1
                continue
            points = shape["points"]
            if len(points) < 3:
                continue
            bbox = polygon_to_bbox(points)
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": LABEL_TO_CAT_ID[label],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "segmentation": [[round(c) for p in points for c in p]],
                "iscrowd": 0,
            })
            ann_id += 1
            inst_counter[label] += 1
            classes_in_img.add(label)

        if not classes_in_img:
            empty_images += 1
        for c in classes_in_img:
            img_with_class[c] += 1

    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": v, "name": k}
                       for k, v in LABEL_TO_CAT_ID.items()],
    }
    ANNOTATION_FILE.write_text(
        json.dumps(coco_data, indent=2, ensure_ascii=False))

    # 统计输出
    print("\n===== 数据统计 =====")
    print(f"标注图片总数: {len(coco_images)}")
    print(f"实例总数: {len(coco_annotations)}")
    for label in LABEL_TO_CAT_ID:
        print(f"  {label:>12}: {inst_counter[label]:>4} 实例 / "
              f"{img_with_class[label]:>4} 张图")
    print(f"空标注图(负样本): {empty_images}")
    if skipped_labels:
        print(f"跳过的未知标签: {dict(skipped_labels)}")
    print(f"Saved: {ANNOTATION_FILE}")


if __name__ == "__main__":
    convert_dataset()
