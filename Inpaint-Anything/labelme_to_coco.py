"""Convert LabelMe JSON annotations to COCO JSON for SAM3 fine-tuning.

输入目录假设: 图片和同名 .json 放在一起，例如
    /path/to/dir/frame_00001.png
    /path/to/dir/frame_00001.json
输出: 一个 COCO JSON 文件，包含 images / annotations / categories。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def labelme_to_coco(input_dir: str, output_json: str):
    input_dir_path = Path(input_dir)
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 先扫描所有 JSON，收集类别名称，保证多类别稳定映射
    label_files = sorted(input_dir_path.glob("*.json"))
    if not label_files:
        raise FileNotFoundError(f"No LabelMe JSON files found in {input_dir_path}")

    all_labels = set()
    for json_path in label_files:
        with open(json_path) as f:
            lm = json.load(f)
        for shape in lm.get("shapes", []):
            label = shape.get("label")
            if label:
                all_labels.add(label)

    # 按字母序分配 category_id，确保可复现
    sorted_labels = sorted(all_labels)
    label_to_id = {label: idx + 1 for idx, label in enumerate(sorted_labels)}

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": cid,
                "name": label,
                "supercategory": label,
            }
            for label, cid in label_to_id.items()
        ],
    }

    print(f"Found {len(sorted_labels)} categories: {sorted_labels}")
    if not label_files:
        raise FileNotFoundError(f"No LabelMe JSON files found in {input_dir_path}")

    ann_id = 1
    for img_id, json_path in enumerate(label_files, 1):
        with open(json_path) as f:
            lm = json.load(f)

        img_name = lm.get("imagePath") or json_path.with_suffix(".png").name
        img_path = input_dir_path / img_name
        if not img_path.exists():
            # 如果 imagePath 是相对子路径或不存在，尝试用 json 文件名对应图片
            img_path = json_path.with_suffix(".png")
            if not img_path.exists():
                img_path = json_path.with_suffix(".jpg")

        with Image.open(img_path) as img:
            width, height = img.size

        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "height": height,
                "width": width,
            }
        )

        for shape in lm.get("shapes", []):
            points = np.asarray(shape["points"], dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3:
                continue

            label = shape.get("label")
            if label not in label_to_id:
                continue
            category_id = label_to_id[label]

            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            area = float(bbox_w * bbox_h)

            # COCO polygon segmentation: 扁平化的 [x1,y1,x2,y2,...]
            segmentation = points.flatten().tolist()

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x_min), float(y_min), float(bbox_w), float(bbox_h)],
                    "segmentation": [segmentation],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Converted {len(label_files)} images, {ann_id - 1} annotations -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to COCO JSON for SAM3 fine-tuning."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/kemove/INNOV/datasets/sam_train/episode_000000",
        help="Directory containing paired image and LabelMe JSON files.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/kemove/INNOV/datasets/sam_train/episode_000000.coco.json",
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--category_name",
        type=str,
        default="forearm",
        help="[Deprecated] Category name is now auto-detected from all LabelMe labels.",
    )
    args = parser.parse_args(sys.argv[1:])
    labelme_to_coco(args.input_dir, args.output_json)


if __name__ == "__main__":
    main()
