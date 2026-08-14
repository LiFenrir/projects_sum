#!/usr/bin/env python3
"""
按 image 维度划分 COCO JSON 为 train/val。

输入: annotations.json (全量)
输出: train_annotations.json (80%) / val_annotations.json (20%)
"""

import argparse
import json
import random
from pathlib import Path


def split_dataset():
    parser = argparse.ArgumentParser(description="按 image 维度划分 COCO JSON")
    parser.add_argument(
        "--input", type=str,
        default="/home/kemove/INNOV/datasets/sam_datasets_v2/annotations.json",
        help="全量 annotations.json 路径")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    INPUT_FILE = Path(args.input)
    OUTPUT_DIR = INPUT_FILE.parent
    TRAIN_FILE = OUTPUT_DIR / "train_annotations.json"
    VAL_FILE = OUTPUT_DIR / "val_annotations.json"
    TRAIN_RATIO = args.train_ratio
    SEED = args.seed
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 收集所有 image_id
    image_ids = sorted(img["id"] for img in coco["images"])
    print(f"Total images: {len(image_ids)}")
    print(f"Total annotations: {len(coco['annotations'])}")
    print(f"Categories: {[c['name'] for c in coco['categories']]}")

    # 按 image_id 划分
    random.seed(SEED)
    random.shuffle(image_ids)

    split_idx = int(len(image_ids) * TRAIN_RATIO)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    print(f"\nTrain images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")

    # 过滤 annotations
    train_annotations = [
        a for a in coco["annotations"] if a["image_id"] in train_ids
    ]
    val_annotations = [
        a for a in coco["annotations"] if a["image_id"] in val_ids
    ]

    print(f"Train instances: {len(train_annotations)}")
    print(f"Val instances: {len(val_annotations)}")

    # 按类别统计
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    for split_name, annotations in [
        ("train", train_annotations),
        ("val", val_annotations),
    ]:
        counts = {}
        for a in annotations:
            name = cat_id_to_name[a["category_id"]]
            counts[name] = counts.get(name, 0) + 1
        print(f"\n{split_name} per class:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")

    # 写入
    for split_name, split_ids, split_annotations in [
        ("train", train_ids, train_annotations),
        ("val", val_ids, val_annotations),
    ]:
        output = {
            "images": [img for img in coco["images"] if img["id"] in split_ids],
            "annotations": split_annotations,
            "categories": coco["categories"],
        }
        output_file = TRAIN_FILE if split_name == "train" else VAL_FILE
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    split_dataset()
